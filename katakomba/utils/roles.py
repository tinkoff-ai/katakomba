"""
Applies to NetHack 3.6.6
Source: https://nethackwiki.com/wiki/Role#Role_table_by_alignment_and_race
"""
import enum

from typing import Tuple


class Role(enum.Enum):
    ARCHEOLOGIST = "arc"
    BARBARIAN = "bar"
    CAVEMAN = "cav"
    HEALER = "hea"
    KNIGHT = "kni"
    MONK = "mon"
    PRIEST = "pri"
    RANGER = "ran"
    ROGUE = "rog"
    SAMURAI = "sam"
    TOURIST = "tou"
    VALKYRIE = "val"
    WIZARD = "wiz"


class Race(enum.Enum):
    HUMAN = "hum"
    ELF = "elf"
    DWARF = "dwa"
    GNOME = "gno"
    ORC = "orc"


class Alignment(enum.Enum):
    NEUTRAL = "neu"
    LAWFUL = "law"
    CHAOTIC = "cha"


class Sex(enum.Enum):
    MALE = "mal"
    FEMALE = "fem"


### These combinations are allowed by NetHack
### On sex: both are always available except Valkyrie (which is always female)
ALLOWED_COMBOS = set(
    [
        (Role.ARCHEOLOGIST, Race.HUMAN, Alignment.LAWFUL),
        (Role.ARCHEOLOGIST, Race.HUMAN, Alignment.NEUTRAL),
        (Role.ARCHEOLOGIST, Race.DWARF, Alignment.LAWFUL),
        (Role.ARCHEOLOGIST, Race.GNOME, Alignment.NEUTRAL),
        (Role.BARBARIAN, Race.HUMAN, Alignment.NEUTRAL),
        (Role.BARBARIAN, Race.HUMAN, Alignment.CHAOTIC),
        (Role.BARBARIAN, Race.ORC, Alignment.CHAOTIC),
        (Role.CAVEMAN, Race.HUMAN, Alignment.LAWFUL),
        (Role.CAVEMAN, Race.HUMAN, Alignment.NEUTRAL),
        (Role.CAVEMAN, Race.DWARF, Alignment.LAWFUL),
        (Role.CAVEMAN, Race.GNOME, Alignment.NEUTRAL),
        (Role.HEALER, Race.HUMAN, Alignment.NEUTRAL),
        (Role.HEALER, Race.GNOME, Alignment.NEUTRAL),
        (Role.KNIGHT, Race.HUMAN, Alignment.LAWFUL),
        (Role.MONK, Race.HUMAN, Alignment.NEUTRAL),
        (Role.MONK, Race.HUMAN, Alignment.LAWFUL),
        (Role.MONK, Race.HUMAN, Alignment.CHAOTIC),
        (Role.PRIEST, Race.HUMAN, Alignment.NEUTRAL),
        (Role.PRIEST, Race.HUMAN, Alignment.LAWFUL),
        (Role.PRIEST, Race.HUMAN, Alignment.CHAOTIC),
        (Role.PRIEST, Race.ELF, Alignment.CHAOTIC),
        (Role.RANGER, Race.HUMAN, Alignment.NEUTRAL),
        (Role.RANGER, Race.HUMAN, Alignment.CHAOTIC),
        (Role.RANGER, Race.ELF, Alignment.CHAOTIC),
        (Role.RANGER, Race.GNOME, Alignment.NEUTRAL),
        (Role.RANGER, Race.ORC, Alignment.CHAOTIC),
        (Role.ROGUE, Race.HUMAN, Alignment.CHAOTIC),
        (Role.ROGUE, Race.ORC, Alignment.CHAOTIC),
        (Role.SAMURAI, Race.HUMAN, Alignment.LAWFUL),
        (Role.TOURIST, Race.HUMAN, Alignment.NEUTRAL),
        (Role.VALKYRIE, Race.HUMAN, Alignment.NEUTRAL),
        (Role.VALKYRIE, Race.HUMAN, Alignment.LAWFUL),
        (Role.VALKYRIE, Race.DWARF, Alignment.LAWFUL),
        (Role.WIZARD, Race.HUMAN, Alignment.NEUTRAL),
        (Role.WIZARD, Race.HUMAN, Alignment.CHAOTIC),
        (Role.WIZARD, Race.ELF, Alignment.CHAOTIC),
        (Role.WIZARD, Race.GNOME, Alignment.NEUTRAL),
        (Role.WIZARD, Race.ORC, Alignment.CHAOTIC),
    ]
)


def decode_character_str(character: str) -> Tuple[Role, Race, Alignment, Sex]:
    if character.count("-") != 3:
        raise Exception("Cannot decode the character without full specification.")

    settings = character.split("-")
    role = Role._value2member_map_[str.lower(settings[0])]
    race = Race._value2member_map_[str.lower(settings[1])]
    alignment = Alignment._value2member_map_[str.lower(settings[2])]
    sex = Sex._value2member_map_[str.lower(settings[3])]

    return role, race, alignment, sex
