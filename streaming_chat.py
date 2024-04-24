# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
import os
os.environ.update({
    key: value for key, value in map(
        lambda x:x.split("="), "WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=localhost MASTER_PORT=19990".split(" "))
})

from typing import List, Optional
import fire
from llama import Dialog, StreamingLlama

def main(
    ckpt_dir: str="Meta-Llama-3-8B-Instruct/",
    tokenizer_path: str="Meta-Llama-3-8B-Instruct/tokenizer.model",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    generator = StreamingLlama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs: List[Dialog] = [[]]

    while True:
        dialog = dialogs[0]
        content = input("User: ")
        dialog.append(dict(role="user", content=content))

        assistant_output = ""
        for token, word in generator.chat_completion(
                dialogs,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            ):
            assistant_output += word
            print(word, end="", flush=True)
        
        print("\n", end="", flush=True)
        dialog.append(dict(role="assistant", content=assistant_output.strip()))

if __name__ == "__main__":
    fire.Fire(main)
