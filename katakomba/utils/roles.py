"""
Applies to NetHack 3.6.6
Source: https://nethackwiki.com/wiki/Role#Role_table_by_alignment_and_race
"""
import enum

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
### On sex: both are always available except Valkyrie (which has no sex)
ALLOWED_COMBOS = set([
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
])

# These are combinations for the splits from the paper
BASE_COMBOS = set([
    (Role.ARCHEOLOGIST, Race.HUMAN, Alignment.NEUTRAL),
    (Role.CAVEMAN, Race.HUMAN, Alignment.NEUTRAL),
    (Role.BARBARIAN, Race.HUMAN, Alignment.NEUTRAL),
    (Role.HEALER, Race.HUMAN, Alignment.NEUTRAL),
    (Role.KNIGHT, Race.HUMAN, Alignment.LAWFUL),
    (Role.MONK, Race.HUMAN, Alignment.NEUTRAL),
    (Role.PRIEST, Race.HUMAN, Alignment.NEUTRAL),
    (Role.RANGER, Race.HUMAN, Alignment.NEUTRAL),
    (Role.ROGUE, Race.HUMAN, Alignment.CHAOTIC),
    (Role.SAMURAI, Race.HUMAN, Alignment.LAWFUL),
    (Role.TOURIST, Race.HUMAN, Alignment.NEUTRAL),
    (Role.VALKYRIE, Race.HUMAN, Alignment.NEUTRAL),
    (Role.WIZARD, Race.HUMAN, Alignment.NEUTRAL)
])

EXTENDED_COMBOS = set([
    (Role.PRIEST, Race.ELF, Alignment.CHAOTIC),
    (Role.RANGER, Race.ELF, Alignment.CHAOTIC),
    (Role.WIZARD, Race.ELF, Alignment.CHAOTIC),
    (Role.ARCHEOLOGIST, Race.DWARF, Alignment.LAWFUL),
    (Role.CAVEMAN, Race.DWARF, Alignment.LAWFUL),
    (Role.VALKYRIE, Race.DWARF, Alignment.LAWFUL),
    (Role.ARCHEOLOGIST, Race.GNOME, Alignment.NEUTRAL),
    (Role.CAVEMAN, Race.GNOME, Alignment.NEUTRAL),
    (Role.HEALER, Race.GNOME, Alignment.NEUTRAL),
    (Role.RANGER, Race.GNOME, Alignment.NEUTRAL),
    (Role.WIZARD, Race.GNOME, Alignment.NEUTRAL),
    (Role.BARBARIAN, Race.ORC, Alignment.CHAOTIC),
    (Role.RANGER, Race.ORC, Alignment.CHAOTIC),
    (Role.ROGUE, Race.ORC, Alignment.CHAOTIC),
    (Role.WIZARD, Race.ORC, Alignment.CHAOTIC)
])

COMPLETE_COMBOS = set([
    (Role.ARCHEOLOGIST, Race.HUMAN, Alignment.LAWFUL),
    (Role.CAVEMAN, Race.HUMAN, Alignment.LAWFUL),
    (Role.MONK, Race.HUMAN, Alignment.LAWFUL),
    (Role.PRIEST, Race.HUMAN, Alignment.LAWFUL),
    (Role.VALKYRIE, Race.HUMAN, Alignment.LAWFUL),
    (Role.BARBARIAN, Race.HUMAN, Alignment.CHAOTIC),
    (Role.MONK, Race.HUMAN, Alignment.CHAOTIC),
    (Role.PRIEST, Race.HUMAN, Alignment.CHAOTIC),
    (Role.RANGER, Race.HUMAN, Alignment.CHAOTIC),
    (Role.WIZARD, Race.HUMAN, Alignment.CHAOTIC)
])
