import torch
import torch.nn as nn
from torch.nn.parallel import data_parallel
from torch.nn.functional import log_softmax
from reap.utils.tools import batch_sequences
from reap.utils.state import State
from reap.utils.config import PAD
from reap.utils.beam_search import SequenceGenerator


class Seq2Seq(nn.Module):

	def __init__(self, encoder=None, decoder=None, bridge=None):
		super(Seq2Seq, self).__init__()
		self.encoder = encoder
		self.decoder = decoder

		if bridge is not None:
			self.bridge = bridge

	def bridge(self, context):
		return State(context=context,
					 batch_first=getattr(self.decoder, 'batch_first', context.batch_first))

	def encode(self, inputs, input_pos_order = None, hidden=None, device_ids=None):
		if isinstance(device_ids, tuple):
			return data_parallel(self.encoder, (inputs, hidden),
								 device_ids=device_ids,
								 dim=0 if self.encoder.batch_first else 1)
		else:
			return self.encoder(inputs, input_pos_order, hidden)

	def decode(self, inputs, state, get_attention=None, device_ids=None):
		if isinstance(device_ids, tuple):
			inputs = (inputs, state, get_attention) if get_attention else (
				inputs, state)
			return data_parallel(self.decoder, inputs,
								 device_ids=device_ids,
								 dim=0 if self.decoder.batch_first else 1)
		else:
			if get_attention:
				return self.decoder(inputs, state, get_attention=get_attention)
			else:
				return self.decoder(inputs, state)

	def forward(self, input_encoder, input_decoder, input_pos_order, encoder_hidden=None, device_ids=None, get_attention=None):
		if not isinstance(device_ids, dict):
			device_ids = {'encoder': device_ids, 'decoder': device_ids}
		context = self.encode(input_encoder, input_pos_order, encoder_hidden,
							  device_ids=device_ids.get('encoder', None))
		if hasattr(self, 'bridge'):
			state = self.bridge(context)
		output, state = self.decode(
			input_decoder, state, device_ids=device_ids.get('decoder', None), get_attention=get_attention)
		return output, state.attention_score

	def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
	    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
	    https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
	        Args:
	            logits: logits distribution shape (..., vocabulary size)
	            top_k >0: keep only top k tokens with highest probability (top-k filtering).
	            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
	    """
	    top_k = min(top_k, logits.size(-1))  # Safety check
	    if top_k > 0:
	        # Remove all tokens with a probability less than the last token of the top-k
	        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
	        logits[indices_to_remove] = filter_value

	    if top_p > 0.0:
	        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
	        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

	        # Remove tokens with cumulative probability above the threshold
	        sorted_indices_to_remove = cumulative_probs >= top_p
	        # Shift the indices to the right to keep also the first token above the threshold
	        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
	        sorted_indices_to_remove[..., 0] = 0

	        indices_to_remove = torch.zeros_like(logits, dtype=torch.uint8).scatter_(
	            dim=-1, index=sorted_indices, src=sorted_indices_to_remove )
	        logits[indices_to_remove] = filter_value
	    return logits	

	def _decode_step(self, input_list, state_list, k=1,
					 feed_all_timesteps=False,
					 remove_unknown=False,
					 get_attention=False,
					 device_ids=None, 
					 top_p = 0.0, top_k=0):

		view_shape = (-1, 1) if self.decoder.batch_first else (1, -1)
		time_dim = 1 if self.decoder.batch_first else 0
		device = next(self.decoder.parameters()).device

		# For recurrent models, the last input frame is all we care about,
		# use feed_all_timesteps whenever the whole input needs to be fed
		if feed_all_timesteps:
			inputs = [torch.tensor(inp, device=device, dtype=torch.long)
					  for inp in input_list]
			inputs = batch_sequences(
				inputs, device=device, batch_first=self.decoder.batch_first)[0]

		else:
			last_tokens = [inputs[-1] for inputs in input_list]
			inputs = torch.stack(last_tokens).view(*view_shape)

		states = State().from_list(state_list)
		logits, new_states = self.decode(inputs, states,
										 get_attention=get_attention,
										 device_ids=device_ids)
		# use only last prediction
		logits = logits.select(time_dim, -1).contiguous()
		if remove_unknown:
			# Remove possibility of unknown
			logits[:, UNK].fill_(-float('inf'))
		
		if top_k > 0 or top_p > 0:
			logprobs_old = log_softmax(logits, dim=1).squeeze(0)

			logits = self.top_k_top_p_filtering(logits, top_p=top_p, top_k=top_k)
			logprobs = log_softmax(logits, dim=1)
			probs = logprobs.exp()
			words = torch.multinomial(probs, num_samples=k)
			logprobs_ = []
			for w,l in zip(words, logprobs):
				logprobs_.append(l[w])
		else:
			logprobs = log_softmax(logits, dim=1)
			logprobs_, words = logprobs.topk(k, 1)

		new_states_list = [new_states[i] for i in range(len(input_list))]
		return words, logprobs_, new_states_list

	def generate(self, input_encoder, input_decoder, input_order, beam_size=None,
				 max_sequence_length=None, length_normalization_factor=0.0, top_p=0, top_k = 0,
				 get_attention=False, device_ids=None):
		if not isinstance(device_ids, dict):
			device_ids = {'encoder': device_ids, 'decoder': device_ids}
		
		context = self.encode(input_encoder, input_pos_order = input_order,
							  device_ids=device_ids.get('encoder', None))
		if hasattr(self, 'bridge'):
			state = self.bridge(context)
		state_list = state.as_list()
		generator = SequenceGenerator(
			decode_step=self._decode_step,
			beam_size=beam_size,
			max_sequence_length=max_sequence_length,
			get_attention=get_attention,
			length_normalization_factor=length_normalization_factor,
			device_ids=device_ids.get('encoder', None))
		return generator.beam_search(input_decoder, state_list, top_k=top_k, top_p=top_p)