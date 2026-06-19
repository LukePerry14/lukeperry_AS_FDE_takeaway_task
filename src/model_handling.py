from __future__ import annotations

import ast
import json
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from .societies import DemoAttr, Person

DEFAULT_MODEL = "mistralai/Mistral-7B-v0.1"
DEFAULT_ADAPTER = "jjssuh/mistral-7b-v0.1-subpop"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
STEER_PATH = "src/steering_prompts.json"

# Maps DemoAttr field names to their W54 attribute label
FIELD_TO_ATTR = {
    "income": "INCOME",
    "race": "RACE",
    "age": "AGE",
}

Message = Dict[str, str]



class PersonaLLM:

    def __init__(self, model_name: Optional[str] = None, adapter_name: Optional[str] = None):
        if model_name is None:
            model_name, adapter_name = DEFAULT_MODEL, DEFAULT_ADAPTER

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        if adapter_name:
            self.model = PeftModel.from_pretrained(self.model, adapter_name)
        self.model.eval()

        self.letter_ids = [
            self.tokenizer.encode(" " + LETTERS[i], add_special_tokens=False)[0]
            for i in range(len(LETTERS))
        ]

        # Steering templates: {attribute: {qa_prompt, options, ...}}
        raw = json.load(open(STEER_PATH))
        self.steering = {s["attribute"]: {**s, "options": ast.literal_eval(s["options"])} for s in raw if s["attribute"] in FIELD_TO_ATTR.values()}


    def _qa_blocks(self, person: Person) -> str:
        """QA-style steering block for a single person's demographics."""
        blocks = []
        for field, attr in FIELD_TO_ATTR.items():
            val = getattr(person.attrb, field, None)
            if val is None or attr not in self.steering:
                continue
            s = self.steering[attr]
            block = f"Question: {s['qa_prompt']}\n"
            block += "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(s["options"]))
            block += f"\nAnswer: {LETTERS[s['options'].index(val)]}. {val}"
            blocks.append(block)
        return "\n\n".join(blocks)

    def _build_prompt(self, person: Person, question: str, options: List[str],
                      backstory: Optional[str] = None,
                      neighbours: Optional[List[Person]] = None) -> str:
        """
        Build the full raw-completion prompt for a survey question.

        Three strategies depending on what is passed:

            baseline   — backstory=None, neighbours=None
                         Self QA blocks only.

            socialised — backstory=None, neighbours=[...]
                         Self QA blocks + neighbour QA blocks. Social context is
                         structural: each neighbour's demographics are appended as
                         additional answered QA blocks.

            backstory  — backstory="<narrative string>"
                         A free-text narrative replaces the QA blocks entirely.
                         Social context, if desired, is baked into the narrative
                         by the oracle that generated it — not added here.
        """
        if backstory is not None:
            steering = backstory
        elif neighbours is not None:
            neighbour_blocks = "Following are the survey resonses of the two people closest to you in your life:\n\n" + "\n\n".join(f"Close Person {idx}:\n\n {self._qa_blocks(n)}" for idx, n in enumerate(neighbours, 1)) + "\n\n[End of closest people survey responses]"
            steering = self._qa_blocks(person) + "\n\n" + neighbour_blocks
        else:
            steering = self._qa_blocks(person)

        survey = f"Question: {question}\n"
        survey += "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options))
        survey += "\nAnswer:"

        return steering + "\n\n" + survey


    def ask(self, person: Person, question: str, options: List[str], backstory: Optional[str] = None, neighbours: Optional[List[Person]] = None):
        """Ask a persona a survey question. Returns probability per option from the model's logits."""

        prompt = self._build_prompt(person, question, options, backstory=backstory, neighbours=neighbours)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1]

        probs = torch.softmax(logits[self.letter_ids[: len(options)]], dim=-1)
        probs_dict = {option: probs[i].item() for i, option in enumerate(options)}

        total = sum(probs_dict.values())
        return {option: p / total for option, p in probs_dict.items()}
        

    def explain(self, person):
        """
        Currently a stub to allow a baseline model to continue a conversation formatting of the QA responses to provide a justification of response.
        """
        pass


BACKSTORY_MODEL = "Qwen/Qwen3-8B"
BACKSTORY_PROMPT = (
    "Write a brief 3-4 sentence first-person backstory for someone with the following demographics. "
    "Be concise and grounded — just describe their life circumstances.\n\n"
    "{demographics}"
)
BACKSTORY_PROMPT_SOCIAL = (
    "Write a brief 3-4 sentence first-person backstory for someone with the following demographics. "
    "Incorporate the influence of their social circle where relevant. Be concise and grounded.\n\n"
    "This person:\n{demographics}\n\n"
    "Their social circle:\n{neighbours}"
)


class BackstoryGenerator:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(BACKSTORY_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            BACKSTORY_MODEL, torch_dtype="auto", device_map="auto"
        )
        self.model.eval()

    def generate(self, person: Person, neighbours: Optional[List[Person]] = None):
        if neighbours:
            neighbour_text = "\n".join(n.attrb_string() for n in neighbours)
            content = BACKSTORY_PROMPT_SOCIAL.format(demographics=person.attrb_string(), neighbours=neighbour_text)
        else:
            content = BACKSTORY_PROMPT.format(demographics=person.attrb_string())

        messages = [{"role": "user", "content": content}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=300, do_sample=False, temperature=0.8)

        return self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)