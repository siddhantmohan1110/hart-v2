default_prompts = [
    "beautiful lady, freckles, big smile, blue eyes, short ginger hair, dark makeup, wearing a floral blue vest top, soft light, dark grey background",
    "anthropomorphic profile of the white snow owl Crystal priestess , art deco painting, pretty and expressive eyes, ornate costume, mythical, ethereal, intricate, elaborate, hyperrealism, hyper detailed, 3D, 8K, Ultra Realistic, high octane, ultra resolution, amazing detail, perfection, In frame, photorealistic, cinematic lighting, visual clarity, shading , Lumen Reflections, Super-Resolution, gigapixel, color grading, retouch, enhanced, PBR, Blender, V-ray, Procreate, zBrush, Unreal Engine 5, cinematic, volumetric, dramatic, neon lighting, wide angle lens ,no digital painting blur."
    "Bright scene, aerial view, ancient city, fantasy, gorgeous light, mirror reflection, high detail, wide angle lens.",
    "A 4k dslr image of a lemur wearing a red magician hat and a blue coat performing magic tricks with cards in a garden.",
    "A silhouette of a grand piano overlooking a dusky cityscape viewed from a top-floor penthouse, rendered in the bold and vivid sytle of a vintage travel poster.",
    "Crocodile in a sweater.",
    "Luffy from ONEPIECE, handsome face, fantasy.",
    "3d digital art of an adorable ghost, glowing within, holding a heart shaped pumpkin, Halloween, super cute, spooky haunted house background.",
    "an astronaut sitting in a diner, eating fries, cinematic, analog film",
    "Chinese architecture, ancient style,mountain, bird, lotus, pond, big tree, 4K Unity, octane rendering.",
]

llm_system_prompt = """Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation.

Examples:
- User Prompt: A cat sleeping -> A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.
- User Prompt: A busy city street -> A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.

Please generate only the enhanced description for the prompt below and DO NOT include any additional sentences. Start your response with "Enhanced Prompt:".

User Prompt:\n"""

max_seq_len = 10240
max_batch_size = 16

great_prompts = [
    # Group 1 — Landscapes & Nature
    "A misty mountain valley at sunrise with a river flowing through pine forests.",
    "Snow-covered peaks under a purple sunset sky, seen from a wooden cabin window.",
    "A tropical beach with turquoise waters and palm trees swaying in the wind.",
    "A desert landscape with red sand dunes and a caravan of camels at dusk.",

    # # Group 2 — Futuristic & Sci-Fi Cityscapes
    # "A futuristic city skyline with neon lights and flying cars at night.",
    # "A cyberpunk street market filled with holographic signs and robots.",
    # "A floating city above the clouds powered by solar sails and wind turbines.",

    # # Group 3 — Artistic Portraits
    # "A surreal portrait of a woman with galaxies in her eyes and nebula hair.",
    # "An oil painting of an old sailor with deep wrinkles and a weathered gaze.",
    # "A minimalist black-and-white sketch of a young artist painting a mural.",

    # # Group 4 — Fantasy & Mythical Worlds
    # "A dragon soaring over a medieval castle surrounded by glowing runes.",
    # "An elven forest village with bioluminescent plants and crystal streams.",
    # "A battle between a knight and a shadow creature under a blood moon.",

    # # Group 5 — Abstract & Conceptual Art
    # "A geometric abstraction representing time as flowing ribbons of color.",
    # "Fractal patterns resembling a blooming flower made of glass shards.",
    # "An AI-generated dreamscape merging city lights and neural networks.",

    # # Group 6 — Realistic Everyday Scenes
    # "A rainy street in Tokyo with reflections of neon signs on wet pavement.",
    # "A farmer’s market in summer with people buying fresh fruits and flowers.",
    # "A cozy café interior with sunlight streaming through dusty windows.",
    # "A parked vintage car beside a diner under warm evening light."
]

