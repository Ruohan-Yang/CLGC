import os

# 5.6.1 Within-Network Layer Consistency
os.system("python CLGC_main.py --Eval_layers Aarhus_1 Aarhus_2")
os.system("python CLGC_main.py --Eval_layers Aarhus_1 Aarhus_3")
os.system("python CLGC_main.py --Eval_layers Aarhus_1 Aarhus_4")
os.system("python CLGC_main.py --Eval_layers Aarhus_1 Aarhus_5")
os.system("python CLGC_main.py --Eval_layers Aarhus_2 Aarhus_3")
os.system("python CLGC_main.py --Eval_layers Aarhus_2 Aarhus_4")
os.system("python CLGC_main.py --Eval_layers Aarhus_2 Aarhus_5")
os.system("python CLGC_main.py --Eval_layers Aarhus_3 Aarhus_4")
os.system("python CLGC_main.py --Eval_layers Aarhus_3 Aarhus_5")
os.system("python CLGC_main.py --Eval_layers Aarhus_4 Aarhus_5")

os.system("python CLGC_main.py --Eval_layers Kapferer_1 Kapferer_2")
os.system("python CLGC_main.py --Eval_layers Kapferer_1 Kapferer_3")
os.system("python CLGC_main.py --Eval_layers Kapferer_1 Kapferer_4")
os.system("python CLGC_main.py --Eval_layers Kapferer_2 Kapferer_3")
os.system("python CLGC_main.py --Eval_layers Kapferer_2 Kapferer_4")
os.system("python CLGC_main.py --Eval_layers Kapferer_3 Kapferer_4")

os.system("python CLGC_main.py --Eval_layers Enron_1 Enron_2")

os.system("python CLGC_main.py --Eval_layers LonRail_1 LonRail_2")
os.system("python CLGC_main.py --Eval_layers LonRail_2 LonRail_3")
os.system("python CLGC_main.py --Eval_layers LonRail_1 LonRail_3")

# 5.6.2 Pairwise Cross-Network  Layer Consistency
os.system("python CLGC_main.py --Eval_layers Aarhus_1 Enron_1")
os.system("python CLGC_main.py --Eval_layers Aarhus_1 LonRail_1")
os.system("python CLGC_main.py --Eval_layers Aarhus_1 Kapferer_1")
os.system("python CLGC_main.py --Eval_layers Enron_1 LonRail_1")
os.system("python CLGC_main.py --Eval_layers Enron_1 Kapferer_1")
os.system("python CLGC_main.py --Eval_layers LonRail_1 Kapferer_1")


# 5.6.3 Groupwise Cross-Network Layer Consistency.
os.system("python CLGC_main.py --Eval_layers Aarhus_2 Enron_2 LonRail_2")
os.system("python CLGC_main.py --Eval_layers Enron_2 Kapferer_2 LonRail_2")
os.system("python CLGC_main.py --Eval_layers Aarhus_2 Enron_2 Kapferer_2")
os.system("python CLGC_main.py --Eval_layers Aarhus_2 Kapferer_2 LonRail_2")
os.system("python CLGC_main.py --Eval_layers Aarhus_3 Enron_2 Kapferer_3 LonRail_3")
os.system("python CLGC_main.py --Eval_layers Aarhus_2 Enron_2 Kapferer_2 LonRail_2")
os.system("python CLGC_main.py --Eval_layers Aarhus_1 Enron_1 Kapferer_1 LonRail_1")

# 5.6.4 Theoretical Network Comparison
os.system("python CLGC_main.py --Eval_layers Aarhus_1 random_graph")
os.system("python CLGC_main.py --Eval_layers Aarhus_1 small_world")
os.system("python CLGC_main.py --Eval_layers Aarhus_1 scale_free")
os.system("python CLGC_main.py --Eval_layers Enron_2 random_graph")
os.system("python CLGC_main.py --Eval_layers Enron_2 small_world")
os.system("python CLGC_main.py --Eval_layers Enron_2 scale_free")

print('finish')
