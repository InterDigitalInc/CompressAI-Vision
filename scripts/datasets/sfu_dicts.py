# From m59143


## Dicts for SFU-HW sequences
seq_dict = {
    "Traffic": ("A", "Traffic_2560x1600_30_val"),
    "Kimono": ("B", "Kimono_1920x1080_24_val"),
    "ParkScene": ("B", "ParkScene_1920x1080_24_val"),
    "Cactus": ("B", "Cactus_1920x1080_50_val"),
    "BasketballDrive": ("B", "BasketballDrive_1920x1080_50_val"),
    "BQTerrace": ("B", "BQTerrace_1920x1080_60_val"),
    "BasketballDrill": ("C", "BasketballDrill_832x480_50_val"),
    "BQMall": ("C", "BQMall_832x480_60_val"),
    "PartyScene": ("C", "PartyScene_832x480_50_val"),
    "RaceHorsesC": ("C", "RaceHorses_832x480_30_val"),
    "BasketballPass": ("D", "BasketballPass_416x240_50_val"),
    "BQSquare": ("D", "BQSquare_416x240_60_val"),
    "BlowingBubbles": ("D", "BlowingBubbles_416x240_50_val"),
    "RaceHorses": (
        "D",
        "RaceHorses_416x240_30_val",
    ),  # Dropped trailing 'D' (for consistency with framework naming)
    # "FourPeople"      : ("E", "FourPeople_1280x720_60_val"), # Dropping class 'E' (unsuitable for experiments)
    # "Johnny"          : ("E", "Johnny_1280x720_60_val"),
    # "KristenAndSara"  : ("E", "KristenAndSara_1280x720_60_val"),
}
res_dict = {
    "A": (2560, 1600),
    "B": (1920, 1080),
    "C": (832, 480),
    "D": (416, 240),
    "E": (1280, 720),
}

fr_dict = {  # (IntraPeriod, FrameRate, FramesToBeEncoded, FrameSkip)
    "Traffic": (32, 30, 33, 117),
    "Kimono": (32, 24, 33, 207),
    "ParkScene": (32, 24, 33, 207),
    "Cactus": (64, 50, 97, 403),
    "BQTerrace": (64, 60, 129, 471),
    "BasketballDrive": (64, 50, 97, 403),
    "BQMall": (64, 60, 129, 471),
    "BasketballDrill": (64, 50, 97, 403),
    "PartyScene": (64, 50, 97, 403),
    "RaceHorsesC": (32, 30, 65, 235),
    "BQSquare": (64, 60, 129, 471),
    "BasketballPass": (64, 50, 97, 403),
    "BlowingBubbles": (64, 50, 97, 403),
    "RaceHorses": (32, 30, 65, 235),  # Dropped trailing 'D'
    "KristenAndSara": (64, 60, 129, 471),
    "Johnny": (64, 60, 129, 471),
    "FourPeople": (64, 60, 129, 471),
}
