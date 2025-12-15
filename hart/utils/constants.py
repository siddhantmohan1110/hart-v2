default_prompts = [
    "goldfish",
    "jack-o'-lantern",
    "head cabbage",
    "cauliflower",
]

llm_system_prompt_old = """Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation.

Examples:
- User Prompt: A cat sleeping -> A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.
- User Prompt: A busy city street -> A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.

Please generate only the enhanced description for the prompt below and DO NOT include any additional sentences. Start your response with "Enhanced Prompt:".

User Prompt:\n"""

llm_system_prompt = """You are given a label from ImageNet Classification Dataset. Some labels like Black widow might be ambiguous. Infer to the right meaning from ImageNet class label and generate the image prompt describing the correct visual attributes of the label. User Prompt:\n"""

max_seq_len = 10240
max_batch_size = 16

summarization_prompt_template = """You are generating ONE text-to-image prompt to be used directly by an image generation model.

Write the prompt as if you want the image to be generated, not described. Do NOT describe a list, do NOT mention prompts, summaries, clusters, or collections.

The value must be a single sentence image-generation prompt.

Strictly avoid meta or generic phrasing.

Banned content (must not appear anywhere): prompt, prompts, answer, inference, summary, cluster, collection, various, depicting, output, to generate, the image should, diverse, images, visual similarity, objects, subject, scene showing, categories

Task:
From the prompts below, infer the most plausible shared visual concept and write ONE concise, visually grounded image-generation prompt.
        
Guidelines:
- Focus on concrete visual attributes: object type, shape, texture, material, color, typical pose or viewpoint, and a likely environment if applicable.
- Capture what is common across the prompts; ignore rare, weak, or incoherent outliers.
- If the prompts span unrelated categories, choose ONE dominant and visually distinctive subject and ignore the rest.
- Do NOT list or reference individual class names.
- Avoid abstract, symbolic, or non-visual language.
- Avoid stylistic adjectives unless clearly implied.

Hard Constraints:
- Output exactly ONE sentence.
- Output must be 30 tokens or fewer.
- No bullet points, lists, quotes, or line breaks.
- No explanations or commentary.
- Produce exactly ONE prompt and nothing else.

Prompts:
{prompts_text}

Prompt:"""

enrichment_prompt_template = """
    You are an expert visual prompt engineer for the HART (Hybrid Autoregressive Transformer) image generation model. Your goal is to convert short, ambiguous ImageNet class labels into rich, unambiguous, photorealistic visual descriptions.

    CRITICAL INSTRUCTION:
    The input labels come from the ImageNet dataset, which is based on the WordNet hierarchy. You must ALWAYS prioritize the WordNet definition of the object.
    - If the label is "Black Widow", you must describe the spider (Latrodectus), NEVER the Marvel character.
    - If the label is "Crane", you must check the context or provide a specific description of the bird (Gruidae) or the construction machine, but default to the most common ImageNet biological class if unsure, or specify the biological distinctiveness.
    - If the label is "Jaguar", describe the cat (Panthera onca), not the car (unless specified).

    Your output format for every label must be:
    [Subject Description] + [Environment/Context] + [Lighting/Style] 

    GUIDELINES:
    1. Subject: Explicitly describe the visual features (color, texture, shape). Use scientific names if helpful for clarity.
    2. Context: Place the object in its natural habitat or typical setting.
    3. Style: Use high-quality keywords (4k, detailed texture, cinematic lighting) to ensure HART generates a high-fidelity image.
    4. Output: Provide ONLY the final prompt text. Do not output conversational filler like "Here is the prompt." 
    5. Output must be 40 tokens or fewer

    Example Input: "Black Widow"
    Example Output: A close-up macro photograph of a Latrodectus spider, commonly known as a black widow, featuring a shiny black bulbous abdomen with a distinctive red hourglass marking. The spider is resting on a chaotic silk web in a dark, shadowy corner. Natural lighting, high contrast, 8k resolution, photorealistic texture.

    Input: {pr} 
    Output: 
    """

banned_meta_terms = [
    "prompt",
    "prompts",
    "answer",
    "inference",
    "summary",
    "cluster",
    "collection",
    "various",
    "depicting",
    "output",
    "to generate",
    "the image should",
    "diverse",
    "images",
    "visual similarity",
    "objects",
    "subject",
    "scene showing",
    "categories",
]
