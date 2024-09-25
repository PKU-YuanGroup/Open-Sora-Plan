import dataclasses
from enum import auto, Enum
from typing import Dict, List, Tuple, Union


class SeparatorStyle(Enum):
    """Different separator style."""
    TWO = auto()
    MPT = auto()


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = "{system_message}"
    # The system message
    system_message: str = ""
    # The names of two roles
    roles: Tuple[str] = ("USER", "ASSISTANT")
    # All messages. Each item is (role, message).
    messages: Tuple = ()
    # The number of few shot examples
    offset: int = 0
    # The separator style and configurations
    sep_style: SeparatorStyle = SeparatorStyle.TWO
    sep: str = "\n"
    sep2: str = None
    version: str = "Unknown"
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if isinstance(message, tuple):
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.MPT:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    if isinstance(message, tuple):
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override and template.name in conv_templates:
        raise AssertionError(f"{template.name} has been registered.")

    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    try:
        return conv_templates[name].copy()
    except KeyError as e:
        raise KeyError(f"Conversation template '{name}' not found.") from e


register_conv_template(
    Conversation(
        name="internlm2-chat",
        system_template="<|im_start|>system\n{system_message}",
        # note: The new system prompt was not used here to avoid changes in benchmark performance.
        # system_message="我是书生·万象，英文名是InternVL，是由上海人工智能实验室及多家合作单位联合开发的多模态大语言模型。",
        system_message="你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。",
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>",
        stop_token_ids=[
            2,
            92543,
            92542
        ]
    )
)


register_conv_template(
    Conversation(
        name="llava-v1",
        system_message="A chat between a curious user and an artificial intelligence assistant. "
               "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("USER", "ASSISTANT"),
        offset=0,
        sep_style=SeparatorStyle.TWO,
        sep=" ",
        sep2="</s>"
    )
)
