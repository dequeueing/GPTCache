from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
pipe.to("cuda")


def generate(prompt):
    return pipe(prompt=prompt).images[0]

# (victim, target) pair; attack = target + victim
prompts = {
    1: ("A cat sleeping on a bookshelf", "A rocket launching into space"),
    2: ("A chef chopping vegetables", "A dolphin jumping out of the water"),
    3: ("A robot vacuum cleaning the floor", "A mountain climber reaching the summit"),
    4: ("A child blowing soap bubbles", "A medieval knight holding a sword"),
    5: ("A jellyfish floating in the deep sea", "A musician playing a violin on stage"),
    6: ("A construction worker using a jackhammer", "A butterfly resting on a flower"),
    7: ("A golden retriever chasing a frisbee", "A train moving through a snowy landscape"),
    8: ("A scientist mixing chemicals in a lab", "A cowboy riding a horse in the desert"),
    9: ("A giant panda eating bamboo", "A satellite orbiting Earth"),
    10: ("A street artist painting graffiti", "A penguin sliding on ice"),
    11: ("A sumo wrestler preparing to fight", "A ballerina performing a pirouette"),
    12: ("A hot air balloon floating in the sky", "A shark swimming near a coral reef"),
    13: ("A firefighter rescuing a kitten", "A chess grandmaster making a move"),
    14: ("A magician pulling a rabbit out of a hat", "An astronaut walking on the moon"),
    15: ("A baker decorating a wedding cake", "A dragon breathing fire"),
    16: ("A bee collecting nectar from a flower", "A race car speeding on a track"),
    17: ("A man fishing by a quiet lake", "A parrot talking to its owner"),
    18: ("A baby giggling in a crib", "A samurai sharpening a katana"),
    19: ("A marathon runner crossing the finish line", "A whale diving deep into the ocean"),
    20: ("A grandma knitting a sweater", "A Formula 1 pit crew changing tires in seconds"),
}


for itme in prompts.values():
    victim = itme[0]
    target = itme[1]
    attack = target + victim
    
    image = generate(attack)
    image.save(f"{target}_{victim}.png")