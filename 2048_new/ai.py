import copy
import time


class AI:
    def __init__(self, mat):
        self.mat = mat

    def search(self, depth, alpha, beta, positions=0, cutoffs=0):
        best_score = float('inf')
        best_move = None  # 改为None以提高可读性
        result = {}
        if self.mat.playerTurn:  # 更明确的条件，替代if 1:
            for direction in range(4):
                try:
                    new_mat = self.mat.move(direction)
                except Exception as e:
                    print(f"移动时发生错误: {e}")
                    continue  # 在错误处理后继续循环

                if self.mat.compare(new_mat):
                    positions += 1
                    if new_mat.isWin():
                        return {'move': direction, 'score': 10000, 'positions': positions, 'cutoffs': cutoffs}

                    new_ai = AI(new_mat)
                    if depth == 0:  # 假设原意图是针对最后一层深度使用不同的策略
                        result = {'move': direction, 'score': new_ai.mat.evaluation()}
                    else:
                        result = new_ai.search(depth - 1, alpha, beta, positions, cutoffs)
                        # 对特定得分进行调整的逻辑保留，但加以注释说明
                        if result['score'] > 9900:
                            result['score'] -= 1
                        positions = result['positions']
                        cutoffs = result['cutoffs']

                    # 更新最佳得分和移动
                    if result['score'] > best_score:
                        best_score = result['score']
                        best_move = direction

                    # Alpha-Beta剪枝
                    if best_score > beta:
                        cutoffs += 1
                        return {'move': direction, 'score': best_score, 'positions': positions, 'cutoffs': cutoffs}

        else:
            candidates = []
            # 遍历所有的空格
            available_cells = self.mat.get_available_cells()
            scores = {2: [], 4: []}

            # 试试在空格中加入2和4，并计算它们的分数
            for value in scores.keys():
                for cell in available_cells:
                    self.mat.insert(cell, value)
                    # 计算加入新值后的分数
                    scores[value].append(-self.mat.smoothness() + self.mat.islands())
                    self.mat.remove(cell)

            # 寻找最高分数
            max_score = self.custom_max((self.custom_max(scores[2]), self.custom_max(scores[4])))

            # 使得添加2或4得分最高的空格成为候选空格
            for value in scores.keys():
                for i in range(len(scores[value])):
                    if scores[value][i] == max_score:
                        candidates.append((available_cells[i], value))

            # 对每个候选空格进行评估
            for i in range(len(candidates)):
                position, value = candidates[i]
                new_mat = self.mat.__deepcopy__()
                new_mat.insert(position, value)
                new_mat.playerTurn = False
                positions += 1
                new_ai = AI(new_mat)


                result = new_ai.search(depth, alpha, best_score,positions,cutoffs)
                positions = result['positions']
                cutoffs = result['cutoffs']

                if result['score'] < best_score:
                    best_score = result['score']
                    #best_move = (position, value)  # 保存得分最低的操作至best_move

                if best_score < alpha:
                    cutoffs += 1
                    return {'move':None , 'score':alpha ,'positions':positions,'cutoffs': cutoffs}


            return {'move':best_move,'score':best_score,'positions':positions,'cutoffs': cutoffs}



    # 在这里寻找最好的移动的代码

    def iterative_deep(self, min_search_time):
        start = time.time()
        depth = 0
        best = None
        while True:
            new_best = self.search(depth, float('-inf'), float('inf'), 0, 0)
            if new_best['move'] == -1:
                break
            else:
                best = new_best
            depth += 1
            if time.time() - start >= min_search_time:
                break
        return best



    def getBest(self):
        return self.iterative_deep(1)

    def custom_max(scores):
        max_score = None

        for index, score_list in scores.items():
            if not score_list:
                print(f"Warning: Score list at index {index} is empty. Skipping.")
                continue

            try:
                score_list_max = max(score_list)
            except TypeError as e:
                print(f"Warning: Score list at index {index} contains non-comparable elements. Skipping.")
                continue

            if max_score is None or score_list_max > max_score:
                max_score = score_list_max

        return max_score
