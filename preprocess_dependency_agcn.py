FULL_MODEL = '/Users/baichuanyang/PycharmProjects/CB/agcn/stanford-corenlp-full-2018-02-27'
punctuation = ['.', ',', ':', '?', '!', '(', ')', '"', '[', ']', ';', '#']
chunk_pos = ['NP', 'PP', 'VP', 'ADVP', 'SBAR', 'ADJP', 'PRT', 'INTJ', 'CONJP', 'LST']

class DepInstanceParser():
    def __init__(self, basicDependencies, tokens=[]):
        self.basicDependencies = basicDependencies
        self.tokens = tokens
        self.words = []
        self.dep_governed_info = []
        self.dep_parsing()

    def dep_parsing(self):
        if len(self.tokens) > 0:
            words = []
            for token in self.tokens:
                words.append(self.change_word(token))
            dep_governed_info = [
                {"word": word}
                for i,word in enumerate(words)
            ]
            self.words = words
        else:
            dep_governed_info = [{}] * len(self.basicDependencies)
        for dep in self.basicDependencies:
            dependent_index = dep['dependent'] - 1
            governed_index = dep['governor'] - 1
            dep_governed_info[dependent_index] = {
                "governor": governed_index,
                "dep": dep['dep']
            }
        self.dep_governed_info = dep_governed_info
        # print(dep_governed_info)

    def change_word(self, word):
        if "-RRB-" in word:
            return word.replace("-RRB-", ")")
        if "-LRB-" in word:
            return word.replace("-LRB-", "(")
        return word

    def get_init_dep_matrix(self):
        dep_adj_matrix = [[0] * len(self.words) for _ in range(len(self.words))]
        dep_type_matrix = [["none"] * len(self.words) for _ in range(len(self.words))]
        for i in range(len(self.words)):
            dep_adj_matrix[i][i] = 1
            dep_type_matrix[i][i] = "self_loop"
        return dep_adj_matrix, dep_type_matrix

    def get_first_order(self, direct=False):
        dep_adj_matrix, dep_type_matrix = self.get_init_dep_matrix()

        for i, dep_info in enumerate(self.dep_governed_info):
            governor = dep_info["governor"]
            dep_type = dep_info["dep"]
            dep_adj_matrix[i][governor] = 1
            dep_adj_matrix[governor][i] = 1
            dep_type_matrix[i][governor] = dep_type if direct is False else "{}_in".format(dep_type)
            dep_type_matrix[governor][i] = dep_type if direct is False else "{}_out".format(dep_type)
        return dep_adj_matrix, dep_type_matrix

    def get_next_order(self, dep_adj_matrix, dep_type_matrix):
        new_dep_adj_matrix = copy.deepcopy(dep_adj_matrix)
        new_dep_type_matrix = copy.deepcopy(dep_type_matrix)
        for target_index in range(len(dep_adj_matrix)):
            for first_order_index in range(len(dep_adj_matrix[target_index])):
                if dep_adj_matrix[target_index][first_order_index] == 0:
                    continue
                for second_order_index in range(len(dep_adj_matrix[first_order_index])):
                    if dep_adj_matrix[first_order_index][second_order_index] == 0:
                        continue
                    if second_order_index == target_index:
                        continue
                    if new_dep_adj_matrix[target_index][second_order_index] == 1:
                        continue
                    new_dep_adj_matrix[target_index][second_order_index] = 1
                    new_dep_type_matrix[target_index][second_order_index] = dep_type_matrix[first_order_index][second_order_index]
        return new_dep_adj_matrix, new_dep_type_matrix

    def get_second_order(self, direct=False):
        dep_adj_matrix, dep_type_matrix = self.get_first_order(direct=direct)
        return self.get_next_order(dep_adj_matrix, dep_type_matrix)

    def get_third_order(self, direct=False):
        dep_adj_matrix, dep_type_matrix = self.get_second_order(direct=direct)
        return self.get_next_order(dep_adj_matrix, dep_type_matrix)

    def search_dep_path(self, start_idx, end_idx, adj_max, dep_path_arr):
        for next_id in range(len(adj_max[start_idx])):
            if next_id in dep_path_arr or adj_max[start_idx][next_id] in ["none"]:
                continue
            if next_id == end_idx:
                return 1, dep_path_arr + [next_id]
            stat, dep_arr = self.search_dep_path(next_id, end_idx, adj_max, dep_path_arr + [next_id])
            if stat == 1:
                return stat, dep_arr
        return 0, []

    def get_dep_path(self, start_range, end_range, direct=False):
        dep_path_adj_matrix, dep_path_type_matrix = self.get_init_dep_matrix()

        first_order_dep_adj_matrix, first_order_dep_type_matrix = self.get_first_order(direct=direct)
        for start_index in start_range:
            for end_index in end_range:
                _, dep_path_indexs = self.search_dep_path(start_index, end_index, first_order_dep_type_matrix, [start_index])
                for left_index, right_index in zip(dep_path_indexs[:-1], dep_path_indexs[1:]):
                    dep_path_adj_matrix[start_index][right_index] = 1
                    dep_path_type_matrix[start_index][right_index] = first_order_dep_type_matrix[left_index][right_index]
                    dep_path_adj_matrix[end_index][left_index] = 1
                    dep_path_type_matrix[end_index][left_index] = first_order_dep_type_matrix[right_index][left_index]
        return dep_path_adj_matrix, dep_path_type_matrix

    def get_local_global_graph(self, start_range, end_range, direct=False):
        dep_path_adj_matrix, dep_path_type_matrix = self.get_init_dep_matrix()

        first_order_dep_adj_matrix, first_order_dep_type_matrix = self.get_first_order(direct=direct)
        for start_index in start_range+end_range:
            for next_id in range(len(first_order_dep_type_matrix[start_index])):
                if first_order_dep_type_matrix[start_index][next_id] in ["none"]:
                    continue
                dep_path_adj_matrix[start_index][next_id] = first_order_dep_adj_matrix[start_index][next_id]
                dep_path_type_matrix[start_index][next_id] = first_order_dep_type_matrix[start_index][next_id]
                dep_path_adj_matrix[next_id][start_index] = first_order_dep_adj_matrix[next_id][start_index]
                dep_path_type_matrix[next_id][start_index] = first_order_dep_type_matrix[next_id][start_index]
        for start_index in start_range:
            for end_index in end_range:
                _, dep_path_indexs = self.search_dep_path(start_index, end_index, first_order_dep_type_matrix, [start_index])
                for left_index, right_index in zip(dep_path_indexs[:-1], dep_path_indexs[1:]):
                    dep_path_adj_matrix[left_index][right_index] = 1
                    dep_path_type_matrix[left_index][right_index] = first_order_dep_type_matrix[left_index][right_index]
                    dep_path_adj_matrix[right_index][left_index] = 1
                    dep_path_type_matrix[right_index][left_index] = first_order_dep_type_matrix[right_index][left_index]
        return dep_path_adj_matrix, dep_path_type_matrix

    def get_local_graph(self, start_range, end_range, direct=False):
        dep_path_adj_matrix, dep_path_type_matrix = self.get_init_dep_matrix()

        first_order_dep_adj_matrix, first_order_dep_type_matrix = self.get_first_order(direct=direct)
        for start_index in start_range+end_range:
            for next_id in range(len(first_order_dep_type_matrix[start_index])):
                if first_order_dep_type_matrix[start_index][next_id] in ["none"]:
                    continue
                dep_path_adj_matrix[start_index][next_id] = first_order_dep_adj_matrix[start_index][next_id]
                dep_path_type_matrix[start_index][next_id] = first_order_dep_type_matrix[start_index][next_id]
                dep_path_adj_matrix[next_id][start_index] = first_order_dep_adj_matrix[next_id][start_index]
                dep_path_type_matrix[next_id][start_index] = first_order_dep_type_matrix[next_id][start_index]
        return dep_path_adj_matrix, dep_path_type_matrix

    def get_global_graph(self, start_range, end_range, direct=False):
        dep_path_adj_matrix, dep_path_type_matrix = self.get_init_dep_matrix()

        first_order_dep_adj_matrix, first_order_dep_type_matrix = self.get_first_order(direct=direct)
        for start_index in start_range:
            for end_index in end_range:
                _, dep_path_indexs = self.search_dep_path(start_index, end_index, first_order_dep_type_matrix, [start_index])
                for left_index, right_index in zip(dep_path_indexs[:-1], dep_path_indexs[1:]):
                    dep_path_adj_matrix[left_index][right_index] = 1
                    dep_path_type_matrix[left_index][right_index] = first_order_dep_type_matrix[left_index][right_index]
                    dep_path_adj_matrix[right_index][left_index] = 1
                    dep_path_type_matrix[right_index][left_index] = first_order_dep_type_matrix[right_index][left_index]
        return dep_path_adj_matrix, dep_path_type_matrix

def connect_path(matric,tokens):
    path=[0]*len(tokens)
    for i in range(len(matric)):
        for j in range(len(matric)):
            if i==j:
                matric[i][j]=0
    connect_path_tokens=[]
    for i,m in enumerate(matric):
        if 1 in m:
            path[i]=1
    for i,flag in enumerate(path):
        if flag==1:
            connect_path_tokens.append(tokens[i])

    return  connect_path_tokens

if __name__ == '__main__':
    from stanfordcorenlp import StanfordCoreNLP
    nlp = StanfordCoreNLP(FULL_MODEL,lang="en")
    import  json
    with open("val_pubmed_matric.json",'r') as fp:
        json_data=json.load(fp)
    props = {'timeout': '5000000','annotators': 'pos, parse, depparse', 'tokenize.whitespace': 'true' ,
                 'ssplit.eolonly': 'true', 'pipelineLanguage': 'en', 'outputFormat': 'json'}
    for rel in json_data:
        print(rel)
        dep_info=[]
        tokens=[]
        for i,line in enumerate(json_data[rel]):
            flag=0
            tokens=[]
            # line["tokens"]=['Urakusai', 'Nagahide', '(', 'Japanese', ':', '有楽斎', '\u3000', '長秀', ')', ',', 'was', 'a', 'designer', 'of', 'ukiyo', '-', 'e', 'style', 'Japanese', 'woodblock', 'prints', 'who', 'was', 'active', 'from', 'about', '1804', 'to', 'about', '1848', '.']
            for j,token in enumerate(line["tokens"]):
                # if token=='\u3000' or token.strip(' ')=='' or token.strip(' ') == '\n' or token.strip(' ')=='\n\n' or token.strip(' ')=='\n\n\n' or  token.strip(' ')=='\n\n\n\n' or  token.strip(' ')=='\n\n\n\n\n' or token.strip(' ')=='\n\n\n\n\n\n':
                # if token.strip()=='':
                #     flag+=1
                #     continue
                # if flag!=0:
                #     if j in line["h"][2][0]:
                #         line["h"][2][0][line["h"][2][0].index(j)]-=flag
                #     if j in line["t"][2][0]:
                #         line["t"][2][0][line["t"][2][0].index(j)]-=flag
                if token.strip()=='':
                    token='#'
                tokens.append(token)
            text=' '.join(tokens)
            # print(line["h"][0].split(" "),line["t"][0].split(" "))
            # print(tokens)
            entity1=line["h"][0].split(" ")[-1]
            entity2=line["t"][0].split(" ")[-1]
            results=nlp.annotate(text, properties=props)
            dep_info=eval(results)["sentences"][0]['basicDependencies']
            # print(dep_info)
            local_global_matric=DepInstanceParser(dep_info,tokens).get_local_global_graph(line["h"][2][0],line["t"][2][0],True)
            first_matric=DepInstanceParser(dep_info,tokens).get_first_order(True)
            # print(line["h"][0])
            # print(line["t"][0])
            # parser_result=connect_path(matric[0],tokens)
            line["tokens"]=tokens
            line["local_global_matric"]=local_global_matric
            line["first_matric"]=first_matric
            # print(line)
    with open('val_pubmed_matric.json','w') as fp:
        json.dump(json_data,fp)


