"""
Here are the statistics for results and the normalization all around.
"""
from katakomba.utils.roles import Role, Race, Alignment

MEAN_SCORES_AUTOASCEND = {
    (Role.ARCHEOLOGIST, Race.DWARF, Alignment.LAWFUL): 5445.69,
    (Role.ARCHEOLOGIST, Race.GNOME, Alignment.NEUTRAL): 5316.57,
    (Role.ARCHEOLOGIST, Race.HUMAN, Alignment.LAWFUL): 5826.35,
    (Role.ARCHEOLOGIST, Race.HUMAN, Alignment.NEUTRAL): 6636.44,
    (Role.BARBARIAN, Race.HUMAN, Alignment.CHAOTIC): 18228.11,
    (Role.BARBARIAN, Race.HUMAN, Alignment.NEUTRAL): 17836.68,
    (Role.BARBARIAN, Race.ORC, Alignment.CHAOTIC): 17594.38,
    (Role.CAVEMAN, Race.DWARF, Alignment.LAWFUL): 11893.48,
    (Role.CAVEMAN, Race.GNOME, Alignment.NEUTRAL): 10083.06,
    (Role.CAVEMAN, Race.HUMAN, Alignment.LAWFUL): 12462.82,
    (Role.CAVEMAN, Race.HUMAN, Alignment.NEUTRAL): 12113.87,
    (Role.HEALER, Race.GNOME, Alignment.NEUTRAL): 3783.93,
    (Role.HEALER, Race.HUMAN, Alignment.NEUTRAL): 4068.27,
    (Role.KNIGHT, Race.HUMAN, Alignment.LAWFUL): 14137.06,
    (Role.MONK, Race.HUMAN, Alignment.CHAOTIC): 18353.30,
    (Role.MONK, Race.HUMAN, Alignment.LAWFUL): 16091.57,
    (Role.MONK, Race.HUMAN, Alignment.NEUTRAL): 17456.05,
    (Role.PRIEST, Race.ELF, Alignment.CHAOTIC): 7109.35,
    (Role.PRIEST, Race.HUMAN, Alignment.CHAOTIC): 8262.56,
    (Role.PRIEST, Race.HUMAN, Alignment.LAWFUL): 6847.99,
    (Role.PRIEST, Race.HUMAN, Alignment.NEUTRAL): 7732.69,
    (Role.RANGER, Race.ELF, Alignment.CHAOTIC): 9014.18,
    (Role.RANGER, Race.GNOME, Alignment.NEUTRAL): 6965.04,
    (Role.RANGER, Race.HUMAN, Alignment.CHAOTIC): 8378.50,
    (Role.RANGER, Race.HUMAN, Alignment.NEUTRAL): 8067.99,
    (Role.RANGER, Race.ORC, Alignment.CHAOTIC): 7608.48,
    (Role.ROGUE, Race.HUMAN, Alignment.CHAOTIC): 4818.20,
    (Role.ROGUE, Race.ORC, Alignment.CHAOTIC): 4897.69,
    (Role.SAMURAI, Race.HUMAN, Alignment.LAWFUL): 11009.36,
    (Role.TOURIST, Race.HUMAN, Alignment.NEUTRAL): 4211.47,
    (Role.VALKYRIE, Race.DWARF, Alignment.LAWFUL): 23473.61,
    (Role.VALKYRIE, Race.HUMAN, Alignment.LAWFUL): 26103.03,
    (Role.VALKYRIE, Race.HUMAN, Alignment.NEUTRAL): 18624.77,
    (Role.WIZARD, Race.ELF, Alignment.CHAOTIC): 5005.16,
    (Role.WIZARD, Race.GNOME, Alignment.NEUTRAL): 4317.51,
    (Role.WIZARD, Race.HUMAN, Alignment.CHAOTIC): 5316.82,
    (Role.WIZARD, Race.HUMAN, Alignment.NEUTRAL): 5323.48,
    (Role.WIZARD, Race.ORC, Alignment.CHAOTIC): 5016.74,
}

MIN_SCORES_AUTOASCEND = {
    (Role.ARCHEOLOGIST, Race.DWARF, Alignment.LAWFUL): 0.00,
    (Role.ARCHEOLOGIST, Race.GNOME, Alignment.NEUTRAL): 0.00,
    (Role.ARCHEOLOGIST, Race.HUMAN, Alignment.LAWFUL): 2.00,
    (Role.ARCHEOLOGIST, Race.HUMAN, Alignment.NEUTRAL): 0.00,
    (Role.BARBARIAN, Race.HUMAN, Alignment.CHAOTIC): 0.00,
    (Role.BARBARIAN, Race.HUMAN, Alignment.NEUTRAL): 0.00,
    (Role.BARBARIAN, Race.ORC, Alignment.CHAOTIC): 0.00,
    (Role.CAVEMAN, Race.DWARF, Alignment.LAWFUL): 0.00,
    (Role.CAVEMAN, Race.GNOME, Alignment.NEUTRAL): 0.00,
    (Role.CAVEMAN, Race.HUMAN, Alignment.LAWFUL): 0.00,
    (Role.CAVEMAN, Race.HUMAN, Alignment.NEUTRAL): 0.00,
    (Role.HEALER, Race.GNOME, Alignment.NEUTRAL): 0.00,
    (Role.HEALER, Race.HUMAN, Alignment.NEUTRAL): 0.00,
    (Role.KNIGHT, Race.HUMAN, Alignment.LAWFUL): 0.00,
    (Role.MONK, Race.HUMAN, Alignment.CHAOTIC): 0.00,
    (Role.MONK, Race.HUMAN, Alignment.LAWFUL): 7.00,
    (Role.MONK, Race.HUMAN, Alignment.NEUTRAL): 0.00,
    (Role.PRIEST, Race.ELF, Alignment.CHAOTIC): 0.00,
    (Role.PRIEST, Race.HUMAN, Alignment.CHAOTIC): 0.00,
    (Role.PRIEST, Race.HUMAN, Alignment.LAWFUL): 0.00,
    (Role.PRIEST, Race.HUMAN, Alignment.NEUTRAL): 0.00,
    (Role.RANGER, Race.ELF, Alignment.CHAOTIC): 0.00,
    (Role.RANGER, Race.GNOME, Alignment.NEUTRAL): 0.00,
    (Role.RANGER, Race.HUMAN, Alignment.CHAOTIC): 3.00,
    (Role.RANGER, Race.HUMAN, Alignment.NEUTRAL): 0.00,
    (Role.RANGER, Race.ORC, Alignment.CHAOTIC): 3.00,
    (Role.ROGUE, Race.HUMAN, Alignment.CHAOTIC): 0.00,
    (Role.ROGUE, Race.ORC, Alignment.CHAOTIC): 0.00,
    (Role.SAMURAI, Race.HUMAN, Alignment.LAWFUL): 0.00,
    (Role.TOURIST, Race.HUMAN, Alignment.NEUTRAL): 0.00,
    (Role.VALKYRIE, Race.DWARF, Alignment.LAWFUL): 0.00,
    (Role.VALKYRIE, Race.HUMAN, Alignment.LAWFUL): 0.00,
    (Role.VALKYRIE, Race.HUMAN, Alignment.NEUTRAL): 16.00,
    (Role.WIZARD, Race.ELF, Alignment.CHAOTIC): 0.00,
    (Role.WIZARD, Race.GNOME, Alignment.NEUTRAL): 0.00,
    (Role.WIZARD, Race.HUMAN, Alignment.CHAOTIC): 0.00,
    (Role.WIZARD, Race.HUMAN, Alignment.NEUTRAL): 0.00,
    (Role.WIZARD, Race.ORC, Alignment.CHAOTIC): 0.00,
}

MAX_SCORES_AUTOASCEND = {
    (Role.ARCHEOLOGIST, Race.DWARF, Alignment.LAWFUL): 83496.00,
    (Role.ARCHEOLOGIST, Race.GNOME, Alignment.NEUTRAL): 110054.00,
    (Role.ARCHEOLOGIST, Race.HUMAN, Alignment.LAWFUL): 84823.00,
    (Role.ARCHEOLOGIST, Race.HUMAN, Alignment.NEUTRAL): 138103.00,
    (Role.BARBARIAN, Race.HUMAN, Alignment.CHAOTIC): 164446.00,
    (Role.BARBARIAN, Race.HUMAN, Alignment.NEUTRAL): 292342.00,
    (Role.BARBARIAN, Race.ORC, Alignment.CHAOTIC): 164296.00,
    (Role.CAVEMAN, Race.DWARF, Alignment.LAWFUL): 161682.00,
    (Role.CAVEMAN, Race.GNOME, Alignment.NEUTRAL): 142460.00,
    (Role.CAVEMAN, Race.HUMAN, Alignment.LAWFUL): 156966.00,
    (Role.CAVEMAN, Race.HUMAN, Alignment.NEUTRAL): 258978.00,
    (Role.HEALER, Race.GNOME, Alignment.NEUTRAL): 69566.00,
    (Role.HEALER, Race.HUMAN, Alignment.NEUTRAL): 64337.00,
    (Role.KNIGHT, Race.HUMAN, Alignment.LAWFUL): 419154.00,
    (Role.MONK, Race.HUMAN, Alignment.CHAOTIC): 223997.00,
    (Role.MONK, Race.HUMAN, Alignment.LAWFUL): 190783.00,
    (Role.MONK, Race.HUMAN, Alignment.NEUTRAL): 171224.00,
    (Role.PRIEST, Race.ELF, Alignment.CHAOTIC): 83744.00,
    (Role.PRIEST, Race.HUMAN, Alignment.CHAOTIC): 58367.00,
    (Role.PRIEST, Race.HUMAN, Alignment.LAWFUL): 99250.00,
    (Role.PRIEST, Race.HUMAN, Alignment.NEUTRAL): 114269.00,
    (Role.RANGER, Race.ELF, Alignment.CHAOTIC): 66690.00,
    (Role.RANGER, Race.GNOME, Alignment.NEUTRAL): 58137.00,
    (Role.RANGER, Race.HUMAN, Alignment.CHAOTIC): 62599.00,
    (Role.RANGER, Race.HUMAN, Alignment.NEUTRAL): 54874.00,
    (Role.RANGER, Race.ORC, Alignment.CHAOTIC): 69244.00,
    (Role.ROGUE, Race.HUMAN, Alignment.CHAOTIC): 68628.00,
    (Role.ROGUE, Race.ORC, Alignment.CHAOTIC): 54892.00,
    (Role.SAMURAI, Race.HUMAN, Alignment.LAWFUL): 155163.00,
    (Role.TOURIST, Race.HUMAN, Alignment.NEUTRAL): 59484.00,
    (Role.VALKYRIE, Race.DWARF, Alignment.LAWFUL): 1136591.00,
    (Role.VALKYRIE, Race.HUMAN, Alignment.LAWFUL): 428274.00,
    (Role.VALKYRIE, Race.HUMAN, Alignment.NEUTRAL): 313858.00,
    (Role.WIZARD, Race.ELF, Alignment.CHAOTIC): 71664.00,
    (Role.WIZARD, Race.GNOME, Alignment.NEUTRAL): 37376.00,
    (Role.WIZARD, Race.HUMAN, Alignment.CHAOTIC): 55185.00,
    (Role.WIZARD, Race.HUMAN, Alignment.NEUTRAL): 71709.00,
    (Role.WIZARD, Race.ORC, Alignment.CHAOTIC): 40871.00,
}