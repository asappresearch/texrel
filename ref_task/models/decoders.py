from ref_task.models import decoders_differentiable, decoders_reinforce
from ref_task.models.decoders_differentiable import DifferentiableDecoder
from ref_task.models.decoders_reinforce import StochasticDecoder


# def build_decoder(params, use_reinforce: bool, utt_len: int, vocab_size: int):
#     p = params
#     decoders = decoders_reinforce if use_reinforce else decoders_differentiable
#     if p.sender_decoder == 'IdentityDecoder':
#         decoder_params = {}
#     else:
#         decoder_params = {
#             'embedding_size': p.embedding_size,
#             'out_seq_len': utt_len,
#             'vocab_size': vocab_size
#         }
#     Decoder = getattr(decoders, p.sender_decoder)
#     decoder = Decoder(**decoder_params)
#     return decoder


def build_stochastic_decoder(params, utt_len: int, vocab_size: int) -> StochasticDecoder:
    p = params
    if p.sender_decoder == 'IdentityDecoder':
        decoder_params = {}
    else:
        decoder_params = {
            'embedding_size': p.embedding_size,
            'out_seq_len': utt_len,
            'vocab_size': vocab_size
        }
    Decoder = getattr(decoders_reinforce, p.sender_decoder)
    decoder = Decoder(**decoder_params)
    return decoder


def build_differentiable_decoder(
        params, utt_len: int, vocab_size: int) -> DifferentiableDecoder:
    p = params
    if p.sender_decoder == 'IdentityDecoder':
        decoder_params = {}
    else:
        decoder_params = {
            'embedding_size': p.embedding_size,
            'out_seq_len': utt_len,
            'vocab_size': vocab_size
        }
    Decoder = getattr(decoders_differentiable, p.sender_decoder)
    decoder = Decoder(**decoder_params)
    return decoder
