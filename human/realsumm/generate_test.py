import copy
import re
import pickle

def main():
    used_test_id = [1017, 10586, 11343, 1521, 2736, 3789, 5025, 5272, 5576, 6564, 7174, 7770, 8334, 9325, 9781, 10231, 10595, 11351, 1573, 2748, 3906, 5075, 5334, 5626, 6714, 7397, 7823, 8565, 9393, 9825, 10325, 10680, 11355, 1890, 307, 4043, 5099, 5357, 5635, 6731, 7535, 7910, 8613, 9502, 10368, 10721, 1153, 19, 3152, 4303, 5231, 5420, 5912, 6774, 7547, 8001, 8815, 9555, 10537, 10824, 1173, 1944, 3172, 4315, 5243, 5476, 6048, 6784, 7584, 8054, 8997, 9590, 10542, 11049, 1273, 2065, 3583, 4637, 5244, 5524, 6094, 6976, 7626, 8306, 9086, 9605, 10563, 11264, 1492, 2292, 3621, 4725, 5257, 5558, 6329, 7058, 7670, 8312, 9221, 9709]
    cnndm_test_articles = []
    with open("src.txt", "r", encoding="utf-8") as f:
        cnndm_test_articles = list(f)
    
    used_articles = [cnndm_test_articles[i] for i in used_test_id]
    print(len(used_articles))
    
    sd_abs_path = "abs.pkl"
    sd_ext_path = "ext.pkl"
    sd_abs = pickle.load(open(sd_abs_path, "rb"))
    sd_ext = pickle.load(open(sd_ext_path, "rb"))
    sd = copy.deepcopy(sd_abs)
    for doc_id in sd:
        isd_sota_ext = sd_ext[doc_id]
        isd_sota_ext['system_summaries']['bart_out_ext.txt'] = isd_sota_ext['system_summaries']['bart_out.txt']
        sd[doc_id]['system_summaries'].update(isd_sota_ext['system_summaries'])

    with open("realsumm_100.tsv", "w", encoding="utf-8") as f:
        for doc_id in sd:
            # print(sd[doc_id]["doc_id"], doc_id)
            doc_src = used_articles[doc_id]
            doc_src = doc_src.replace("\t", " ")
            doc_src = doc_src.strip()
            # print(doc_src)
            # print(sd[doc_id]["ref_summ"])
            line = [doc_src]
            for sys_name, system in sd[doc_id]["system_summaries"].items():
                sys_sum = system["system_summary"]
                sys_sum = sys_sum.replace("<t>", "")
                sys_sum = sys_sum.replace("</t>", "")
                sys_sum = sys_sum.replace("\t", " ")
                sys_sum = sys_sum.strip()
                line.append(sys_sum)
            
            linestr = "\t".join(line)
            linestr = re.sub(" +", " ", linestr)
            f.write(linestr + "\n")

if __name__ == '__main__':
    main()