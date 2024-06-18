class BaseLlmClient:
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.params = kwargs

    async def inference(self, prompt: str, max_tokens: int = 0):
        raise NotImplementedError
