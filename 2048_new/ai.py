import copy
import time
from functional import  mat


class AI:
    def __init__(self, mat):
        self.mat = mat



    def search(self, depth, alpha, beta, positions=0, cutoffs=0):
        self.best_score = None
        self.best_move = -1
        self.result = {}

        if self.mat.playerTurn:
            self.best_score = alpha
            for direction in range(4):
                try:
                    new_mat = mat(self.mat.move(direction))
                    new_mat.playerTurn = False
                except Exception as e:
                    print(f"移动时发生错误: {e}")
                    continue

                if self.mat.compare(new_mat):
                    positions += 1
                    if new_mat.isWin():
                        return {'move': direction, 'score': 10000, 'positions': positions, 'cutoffs': cutoffs}

                    new_ai = AI(new_mat)
                    if depth == 0:
                        self.result = {'move': direction, 'score': new_ai.mat.evaluation()}
                    else:
                        self.result = new_ai.search(depth - 1, self.best_score, beta, positions, cutoffs)
                        if self.result['score'] > 9900:
                            self.result['score'] -= 1
                        positions = self.result['positions']
                        cutoffs = self.result['cutoffs']

                    if self.result['score'] > self.best_score:
                        self.best_score = self.result['score']
                        self.best_move = direction
                    if self.best_score > beta:  # Alpha-Beta pruning condition
                        cutoffs += 1
                        return {'move': self.best_move, 'score': beta, 'positions': positions, 'cutoffs': cutoffs}

        else:
            self.best_score = beta
            candidates = []
            available_cells = self.mat.get_available_cells()
            scores = {2: [], 4: []}

            for value in scores.keys():
                for cell in available_cells:
                    self.mat.insert(cell, value)
                    scores[value].append(-self.mat.smoothness() + self.mat.islands())
                    self.mat.remove(cell)

            max_score = self.custom_max((self.custom_max(scores[2]), self.custom_max(scores[4])))

            for value in scores.keys():
                for i in range(len(scores[value])):
                    if scores[value][i] == max_score:
                        candidates.append((available_cells[i], value))

            for i in range(len(candidates)):
                position, value = candidates[i]
                new_mat = copy.deepcopy(self.mat)
                new_mat.insert(position, value)
                new_mat.playerTurn = True
                positions += 1
                new_ai = AI(new_mat)

                self.result = new_ai.search(depth, alpha, self.best_score, positions, cutoffs)
                positions = self.result['positions']
                cutoffs = self.result['cutoffs']

                if self.result['score'] < self.best_score:
                    self.best_score = self.result['score']
                if self.best_score < alpha:
                    cutoffs += 1
                    return {'move': None, 'score': alpha, 'positions': positions, 'cutoffs': cutoffs}

        return {'move': self.best_move, 'score': self.best_score, 'positions': positions, 'cutoffs': cutoffs}

    # 在这里寻找最好的移动的代码

    def iterative_deep(self, min_search_time, max_depth=10):
        self.start = time.time()
        self.depth = 0
        self.best = None
        while True:
            new_best = self.search(self.depth, float('-inf'), float('inf'), 0, 0)
            print(new_best,self.depth)
            if new_best['move'] == -1:
                break
            else:
                self.best = new_best
            print(time.time() - self.start)
            self.depth += 1
            if time.time() - self.start >= min_search_time:
                print('结束')
                break
        return self.best

    def getBest(self):
        return self.iterative_deep(1)

    def custom_max(self, scores):
        if not scores:
            print("Warning: The score list is empty.")
            return None

        valid_scores = [score for score in scores if score is not None]
        if not valid_scores:
            print("Warning: All scores are None.")
            return None

        max_score = max(valid_scores)
        return max_score


