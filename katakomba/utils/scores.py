"""
Here are the statistics for results and the normalization all around.
"""
from katakomba.utils.roles import Role, Race, Alignment, Sex


MEAN_SCORES_AUTOASCEND = {
    (Role.ARCHEOLOGIST, Race.DWARF, Alignment.LAWFUL, Sex.FEMALE): 5616.94,
    (Role.ARCHEOLOGIST, Race.DWARF, Alignment.LAWFUL, Sex.MALE): 5272.48,
    (Role.ARCHEOLOGIST, Race.GNOME, Alignment.NEUTRAL, Sex.FEMALE): 5264.35,
    (Role.ARCHEOLOGIST, Race.GNOME, Alignment.NEUTRAL, Sex.MALE): 5370.49,
    (Role.ARCHEOLOGIST, Race.HUMAN, Alignment.LAWFUL, Sex.FEMALE): 5735.29,
    (Role.ARCHEOLOGIST, Race.HUMAN, Alignment.NEUTRAL, Sex.FEMALE): 6274.40,
    (Role.ARCHEOLOGIST, Race.HUMAN, Alignment.LAWFUL, Sex.MALE): 5905.36,
    (Role.ARCHEOLOGIST, Race.HUMAN, Alignment.NEUTRAL, Sex.MALE): 6985.17,
    (Role.BARBARIAN, Race.HUMAN, Alignment.CHAOTIC, Sex.FEMALE): 18242.08,
    (Role.BARBARIAN, Race.HUMAN, Alignment.NEUTRAL, Sex.FEMALE): 18018.39,
    (Role.BARBARIAN, Race.HUMAN, Alignment.CHAOTIC, Sex.MALE): 18214.56,
    (Role.BARBARIAN, Race.HUMAN, Alignment.NEUTRAL, Sex.MALE): 17637.64,
    (Role.BARBARIAN, Race.ORC, Alignment.CHAOTIC, Sex.FEMALE): 17737.06,
    (Role.BARBARIAN, Race.ORC, Alignment.CHAOTIC, Sex.MALE): 17453.80,
    (Role.CAVEMAN, Race.DWARF, Alignment.LAWFUL, Sex.FEMALE): 11818.33,
    (Role.CAVEMAN, Race.DWARF, Alignment.LAWFUL, Sex.MALE): 11971.92,
    (Role.CAVEMAN, Race.GNOME, Alignment.NEUTRAL, Sex.FEMALE): 10018.27,
    (Role.CAVEMAN, Race.GNOME, Alignment.NEUTRAL, Sex.MALE): 10148.41,
    (Role.CAVEMAN, Race.HUMAN, Alignment.LAWFUL, Sex.FEMALE): 12397.52,
    (Role.CAVEMAN, Race.HUMAN, Alignment.NEUTRAL, Sex.FEMALE): 11438.38,
    (Role.CAVEMAN, Race.HUMAN, Alignment.LAWFUL, Sex.MALE): 12532.68,
    (Role.CAVEMAN, Race.HUMAN, Alignment.NEUTRAL, Sex.MALE): 12781.56,
    (Role.HEALER, Race.GNOME, Alignment.NEUTRAL, Sex.FEMALE): 3712.11,
    (Role.HEALER, Race.GNOME, Alignment.NEUTRAL, Sex.MALE): 3855.02,
    (Role.HEALER, Race.HUMAN, Alignment.NEUTRAL, Sex.FEMALE): 4011.92,
    (Role.HEALER, Race.HUMAN, Alignment.NEUTRAL, Sex.MALE): 4125.63,
    (Role.KNIGHT, Race.HUMAN, Alignment.LAWFUL, Sex.FEMALE): 13949.62,
    (Role.KNIGHT, Race.HUMAN, Alignment.LAWFUL, Sex.MALE): 14323.13,
    (Role.MONK, Race.HUMAN, Alignment.CHAOTIC, Sex.FEMALE): 18720.29,
    (Role.MONK, Race.HUMAN, Alignment.LAWFUL, Sex.FEMALE): 15796.80,
    (Role.MONK, Race.HUMAN, Alignment.NEUTRAL, Sex.FEMALE): 17288.62,
    (Role.MONK, Race.HUMAN, Alignment.CHAOTIC, Sex.MALE): 17999.56,
    (Role.MONK, Race.HUMAN, Alignment.LAWFUL, Sex.MALE): 16391.16,
    (Role.MONK, Race.HUMAN, Alignment.NEUTRAL, Sex.MALE): 17635.66,
    (Role.PRIEST, Race.ELF, Alignment.CHAOTIC, Sex.FEMALE): 7131.26,
    (Role.PRIEST, Race.ELF, Alignment.CHAOTIC, Sex.MALE): 7087.80,
    (Role.PRIEST, Race.HUMAN, Alignment.CHAOTIC, Sex.FEMALE): 8098.44,
    (Role.PRIEST, Race.HUMAN, Alignment.LAWFUL, Sex.FEMALE): 7240.06,
    (Role.PRIEST, Race.HUMAN, Alignment.NEUTRAL, Sex.FEMALE): 8093.07,
    (Role.PRIEST, Race.HUMAN, Alignment.CHAOTIC, Sex.MALE): 8424.20,
    (Role.PRIEST, Race.HUMAN, Alignment.LAWFUL, Sex.MALE): 6487.94,
    (Role.PRIEST, Race.HUMAN, Alignment.NEUTRAL, Sex.MALE): 7385.75,
    (Role.RANGER, Race.ELF, Alignment.CHAOTIC, Sex.FEMALE): 9091.29,
    (Role.RANGER, Race.ELF, Alignment.CHAOTIC, Sex.MALE): 8938.37,
    (Role.RANGER, Race.GNOME, Alignment.NEUTRAL, Sex.FEMALE): 6951.24,
    (Role.RANGER, Race.GNOME, Alignment.NEUTRAL, Sex.MALE): 6978.23,
    (Role.RANGER, Race.HUMAN, Alignment.CHAOTIC, Sex.FEMALE): 8216.96,
    (Role.RANGER, Race.HUMAN, Alignment.NEUTRAL, Sex.FEMALE): 8125.21,
    (Role.RANGER, Race.HUMAN, Alignment.CHAOTIC, Sex.MALE): 8543.32,
    (Role.RANGER, Race.HUMAN, Alignment.NEUTRAL, Sex.MALE): 8013.99,
    (Role.RANGER, Race.ORC, Alignment.CHAOTIC, Sex.FEMALE): 7457.63,
    (Role.RANGER, Race.ORC, Alignment.CHAOTIC, Sex.MALE): 7773.70,
    (Role.ROGUE, Race.HUMAN, Alignment.CHAOTIC, Sex.FEMALE): 4968.14,
    (Role.ROGUE, Race.HUMAN, Alignment.CHAOTIC, Sex.MALE): 4668.82,
    (Role.ROGUE, Race.ORC, Alignment.CHAOTIC, Sex.FEMALE): 4719.07,
    (Role.ROGUE, Race.ORC, Alignment.CHAOTIC, Sex.MALE): 5075.83,
    (Role.SAMURAI, Race.HUMAN, Alignment.LAWFUL, Sex.FEMALE): 11023.48,
    (Role.SAMURAI, Race.HUMAN, Alignment.LAWFUL, Sex.MALE): 10995.25,
    (Role.TOURIST, Race.HUMAN, Alignment.NEUTRAL, Sex.FEMALE): 4291.43,
    (Role.TOURIST, Race.HUMAN, Alignment.NEUTRAL, Sex.MALE): 4131.15,
    (Role.VALKYRIE, Race.DWARF, Alignment.LAWFUL, Sex.FEMALE): 23473.61,
    (Role.VALKYRIE, Race.HUMAN, Alignment.LAWFUL, Sex.FEMALE): 26103.03,
    (Role.VALKYRIE, Race.HUMAN, Alignment.NEUTRAL, Sex.FEMALE): 18624.77,
    (Role.WIZARD, Race.ELF, Alignment.CHAOTIC, Sex.FEMALE): 4899.34,
    (Role.WIZARD, Race.ELF, Alignment.CHAOTIC, Sex.MALE): 5108.16,
    (Role.WIZARD, Race.GNOME, Alignment.NEUTRAL, Sex.FEMALE): 4385.83,
    (Role.WIZARD, Race.GNOME, Alignment.NEUTRAL, Sex.MALE): 4249.88,
    (Role.WIZARD, Race.HUMAN, Alignment.CHAOTIC, Sex.FEMALE): 5288.40,
    (Role.WIZARD, Race.HUMAN, Alignment.NEUTRAL, Sex.FEMALE): 5375.74,
    (Role.WIZARD, Race.HUMAN, Alignment.CHAOTIC, Sex.MALE): 5342.34,
    (Role.WIZARD, Race.HUMAN, Alignment.NEUTRAL, Sex.MALE): 5270.05,
    (Role.WIZARD, Race.ORC, Alignment.CHAOTIC, Sex.FEMALE): 4914.63,
    (Role.WIZARD, Race.ORC, Alignment.CHAOTIC, Sex.MALE): 5111.66,
}


def normalize_score_against_bot(
    score: int, role: Role, race: Race, alignment: Alignment, sex: Sex
) -> float:
    return score / MEAN_SCORES_AUTOASCEND[(role, race, alignment, sex)]
