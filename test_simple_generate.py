import asyncio

from genlm.backend import load_model_by_name
from genlm.bytes import BeamParams, ByteBeamState
from genlm.bytes.byte_lm.trie_state import EOS

PROMPT = b"Respond with a single sentence."


async def generate_until_eos():
    llm = load_model_by_name("gpt2", backend="hf")
    eos_byte = llm.byte_vocab[llm.tokenizer.eos_token_id]
    params = BeamParams(K=16, eos_tokens=[eos_byte, b'\n', b'\n\n'], heal=True, verbose=True)

    state = await ByteBeamState.initial(llm, params)

    try:
        state = await state.prefill(PROMPT)
        output = bytearray(PROMPT)

        while True:
            chart = (await state.logp_next()).materialize()
            next_byte = chart.argmax()
            state = await (state.prune() << next_byte)

            if next_byte in [eos_byte, b'\n', b'\n\n']:
                break
            output.append(next_byte)

        return bytes(output)
    finally:
        await state.cleanup()


asyncio.run(generate_until_eos())
