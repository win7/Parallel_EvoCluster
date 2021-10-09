algorithms = [
	"RING",		# 0
	"TREE",		# 1
	"NETA",		# 2
	"NETB",		# 3
	"TORUS",	# 4
	"GRAPH",	# 5
	"SAME",		# 6
	"GOODBAD",	# 7
	"RAND",		# 8
	"SSA"		# 9
]
emigration = ["CLONE", "REMOVE"]
choice_emi = ["BEST", "WORST", "RANDOM"]
choice_imm = ["BEST", "WORST", "RANDOM"]
number_emi_imm = [1, 2, 3, 4, 5]
interval_emi_imm = [1, 2, 4, 6, 8, 10]

dataset_name = [
	"iris", 			# 0
	"yeast", 			# 1
	"website_phishing", # 2
	"liver", 			# 3
	"ecoli", 			# 4
	"heart", 			# 5
	"diagnosis_II",		# 6
	"aggregation",		# 7
	"vary-density"		# 8
]
clusters = [
	3, 
	10, 
	3, 
	2, 
	7, 
	2, 
	2,
	7,
	3
] # number of clusters in dataset
features = [
	(0, 1, 2, 3),
	(1, 2, 3, 4, 5, 6, 7, 8),
	(0, 1, 2, 3, 4, 5, 6, 7, 8),
	(0, 1, 2, 3, 4, 5),
	(0, 1, 2, 3, 4, 5, 6),
	(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
	(0, 1, 2, 3, 4, 5),
	(0, 1),
	(0, 1)
] # number select of columns dataset

seeds = [
	169735477,
	160028434,
	160897947,
	157407246,
	153881302,
	172694171,
	171070236,
	154302761,
	165786948,
	159504387,
	172007822,
	171759537,
	167673018,
	161471032,
	153969729,
	162772019,
	162871815,
	164484920,
	165299547,
	172039163,
	154936578,
	168577700,
	153992657,
	172048626,
	158530753,
	160026451,
	164317733,
	170918034,
	169403955,
	162541554,
	160937381,
	170219188,
	157430629,
	154508394,
	162819603,
	168764208,
	168557415,
	166309796,
	154966946,
	155241744,
	171859107,
	173430800,
	156284381,
	157136719,
	160813250,
	170995803,
	169041299,
	166136032,
	162228293,
	168958481
]