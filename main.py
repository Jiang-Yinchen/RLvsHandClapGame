from abc import ABC, abstractmethod
from math import exp, inf
import random
import time
import json
from os import remove
import joblib
import matplotlib.pyplot as plt
import sys


class AbstractActor(ABC):
    @abstractmethod
    def choose_action(self, state, use_random=True):
        pass


class Foolish(AbstractActor):
    def __init__(self, MOVEMENT_TABLE):
        self.MOVEMENT_TABLE = MOVEMENT_TABLE

    def choose_action(self, state, use_random=True):
        # 蓄水池抽样
        index = 1
        chosen = None
        for i in self.MOVEMENT_TABLE.keys():
            if self.MOVEMENT_TABLE[i]["need"] <= state[0][0] and self.MOVEMENT_TABLE[i]["combo"] <= state[0][1]:
                if random.random() < 1 / index:
                    chosen = i
                index += 1
        return chosen


class Agent(AbstractActor):
    logs = ""
    START_TIME = time.time()

    def __init__(self):
        self.HYPERPARAMETER_DICT = {}
        self.MOVEMENT_TABLE = {
            "生": {"kill": {"零"}, "need": -1, "combo": 0},
            "防": {"kill": set(), "need": 0, "combo": 0},
            "飞": {"kill": set(), "need": 0, "combo": 0},
            "单": {"kill": {"生", "地"}, "need": 1, "combo": 0},
            "双": {"kill": {"生", "一", "地"}, "need": 2, "combo": 0},
            "弯": {"kill": {"单", "双", "镖", "地", "机"}, "need": 1, "combo": 0},
            "刺": {"kill": {"弯", "肥", "地"}, "need": 1, "combo": 0},
            "肥": {"kill": {"防", "飞", "弯"}, "need": 5, "combo": 0},
            "镖": {"kill": {"飞"}, "need": 3, "combo": 0},
            "零": {"kill": {"防"}, "need": 0, "combo": 0},
            "一": {"kill": set(), "need": -1, "combo": 1},
            "二": {"kill": set(), "need": -2, "combo": 2},
            "三": {"kill": set(), "need": -3, "combo": 3},
            "四": {"kill": set(), "need": -4, "combo": 4},
            "五": {"kill": set(), "need": -5, "combo": 5},
            "胡": {"kill": {"生", "厨"}, "need": 1, "combo": 0},
            "菜": {"kill": {"胡"}, "need": 1, "combo": 0},
            "厨": {"kill": {"菜"}, "need": 2, "combo": 0},
            "地": {"kill": {"生", "一", "二", "三", "四", "五"}, "need": 3, "combo": 0},
            "机": {"kill": {"生", "防", "飞", "单", "双", "刺", "肥", "镖", "一", "二", "三", "四", "五", "胡", "菜", "厨", "地"}, "need": 10, "combo": 0}
        }
        self.START_TIME = time.time()
        self.Q_table = {}
        self.game_round = 0  # 总游戏数
        self.total_round = 0  # 总回合数
        self.test_data = [[], []]
        self.path = ""

    def init_q_table_and_configs(self):
        args = sys.argv[1:]
        if len(args) > 2:
            print("Failed when loading arguments")
            exit()
        elif len(args) == 0:
            self.init_q_table()
            self.HYPERPARAMETER_DICT = json.loads(input("Input hyperparameters or press Enter to start training: "))
        elif len(args) == 1:
            self.init_q_table()
            with open(args[0], "r") as f:
                self.HYPERPARAMETER_DICT = json.load(f)
        else:
            if args[0] == ".":
                self.init_q_table()
            else:
                self.Q_table = joblib.load(args[0])
            if args[1] == ".":
                self.HYPERPARAMETER_DICT = json.loads(input("Input hyperparameters or press Enter to start training: "))
            else:
                with open(args[0], "r") as f:
                    self.HYPERPARAMETER_DICT = json.load(f)

    def save_q_table_and_configs(self):
        if self.path != "":
            self.path += "_" + str(TOTAL_GAME_ROUND)
        else:
            self.path = f"model\\Q_table_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}_{TOTAL_GAME_ROUND}.joblib"
        joblib.dump(self.Q_table, self.path, compress=4)
        print(f"Saved Q_table in file {self.path}")
        with open("training-records.txt", "a") as rf:
            rf.write(f"{self.HYPERPARAMETER_DICT}: {self.test_data[1][-1]:.3%}\n")

    @staticmethod
    def blur(state):
        if state[0] == 0:
            state0 = 0
        elif state[0] <= 1:
            state0 = 1
        elif state[0] <= 2:
            state0 = 2
        elif state[0] <= 4:
            state0 = 3
        elif state[0] <= 6:
            state0 = 4
        elif state[0] <= 9:
            state0 = 5
        elif state[0] <= 12:
            state0 = 6
        else:
            state0 = 7
        if state[1] == 0:
            state1 = 0
        elif state[1] <= 1:
            state1 = 1
        elif state[1] <= 2:
            state1 = 2
        elif state[1] <= 4:
            state1 = 3
        else:
            state1 = 4
        return state0, state1

    def init_q_table(self):  # 初始化 Q 表
        print("Start initializing Q_table.")
        cnt_movement = 0
        cnt_state = 0
        # 使用循环变量来纪念 Dijkstra
        for i in range(16):  # 我的”生“数量
            for j in range(8):  # 对方的”生“数量（模糊后）
                for k in range(6):  # 我的连续”生“数量
                    for s in range(5):  # 对方的连续”生“数量（模糊后）
                        if i >= k and j >= s:
                            self.Q_table[((i, k), (j, s))] = {}
                            for t in self.MOVEMENT_TABLE.keys():
                                if self.MOVEMENT_TABLE[t]["need"] <= i and self.MOVEMENT_TABLE[t]["combo"] <= k:
                                    self.Q_table[((i, k), (j, s))].update({t: {"reward": 0, "episode": 0}})
                                    cnt_movement += 1
                            cnt_state += 1
        print(f"Q_table {hex(hash(str(self.Q_table)))} has been initialized with {cnt_state} state(s) and {cnt_movement} movement(s).")

    def get_alpha(self, episode):  # 获取学习率
        ALPHA_0, ALPHA_MIN, DECAY_EPISODES = self.HYPERPARAMETER_DICT["ALPHA_0"], self.HYPERPARAMETER_DICT["ALPHA_MIN"], self.HYPERPARAMETER_DICT["DECAY_EPISODES"]
        if episode < DECAY_EPISODES:
            return ALPHA_0 - (ALPHA_0 - ALPHA_MIN) * episode / DECAY_EPISODES
        return ALPHA_MIN

    def update_q_table(self, old_state, new_state, action, reward):  # 更新 Q 表
        GAMMA = self.HYPERPARAMETER_DICT["GAMMA"]

        self.Q_table[old_state][action]["episode"] += 1
        new_episode_max_reward = max([self.Q_table[new_state][i]["reward"] for i in self.Q_table[new_state].keys()])  # 新一轮的最大奖励
        old = self.Q_table[old_state][action]["reward"]
        self.Q_table[old_state][action]["reward"] = old + self.get_alpha(self.Q_table[old_state][action]["episode"]) * (reward + GAMMA * new_episode_max_reward - old)  # 更新 Q 表
        self.Q_table[old_state][action]["reward"] = max(min(self.Q_table[old_state][action]["reward"], 10), -10)  # 限制奖励范围

    def get_epsilon(self, rounds):  # 获取探索率
        EPSILON_START, EPSILON_END, EPSILON_DECAY = self.HYPERPARAMETER_DICT["EPSILON_START"], self.HYPERPARAMETER_DICT["EPSILON_END"], self.HYPERPARAMETER_DICT["EPSILON_DECAY"]
        return EPSILON_END + (EPSILON_START - EPSILON_END) * exp(-1 * rounds * EPSILON_DECAY)

    def choose_action(self, state, use_random=True):  # 选择动作
        movements = self.Q_table[state]
        max_reward = -inf
        action = None
        if random.random() >= self.get_epsilon(self.total_round) or not use_random:
            for i in movements.keys():
                i_reward = movements[i]["reward"]
                if i_reward >= max_reward:
                    max_reward = i_reward
                    action = i
        else:
            action = random.choice(list(movements.keys()))
        return action

    @staticmethod
    def judge(player_a_state, player_b_state, player_a_action, player_b_action, MOVEMENT_TABLE):
        now_reward_a, now_reward_b = 0, 0
        ended = 0
        if player_b_action in MOVEMENT_TABLE[player_a_action]["kill"]:
            now_reward_a += 5
            now_reward_b -= 5
            ended = 1
        elif player_a_action in MOVEMENT_TABLE[player_b_action]["kill"]:
            now_reward_a -= 5
            now_reward_b += 5
            ended = -1
        else:
            if MOVEMENT_TABLE[player_a_action]["combo"] != 0:
                player_a_state = (player_a_state[0], 0)
            elif MOVEMENT_TABLE[player_a_action]["need"] > 0:
                player_a_state = (player_a_state[0], 0)
            elif MOVEMENT_TABLE[player_a_action]["need"] != 0:
                player_a_state = (player_a_state[0], min(player_a_state[1] + 1, 5))
            player_a_state = (min(player_a_state[0] - MOVEMENT_TABLE[player_a_action]["need"], 15), player_a_state[1])
            if MOVEMENT_TABLE[player_b_action]["combo"] != 0:
                player_b_state = (player_b_state[0], 0)
            elif MOVEMENT_TABLE[player_b_action]["need"] > 0:
                player_b_state = (player_b_state[0], 0)
            elif MOVEMENT_TABLE[player_b_action]["need"] != 0:
                player_b_state = (player_b_state[0], min(player_b_state[1] + 1, 5))
            player_b_state = (min(player_b_state[0] - MOVEMENT_TABLE[player_b_action]["need"], 15), player_b_state[1])
        return ended, player_a_state, player_b_state, now_reward_a, now_reward_b

    def play_round(self):  # 开始一轮游戏
        state_a, state_b = (0, 0), (0, 0)
        flag = 0
        round_cnt = 0
        while flag == 0:  # 循环直到结束一轮游戏
            old_state_a, old_state_b = state_a, state_b

            action_for_a, action_for_b = self.choose_action((state_a, Agent.blur(state_b))), self.choose_action((state_b, Agent.blur(state_a)))
            flag, state_a, state_b, now_reward_a, now_reward_b = self.judge(state_a, state_b, action_for_a, action_for_b, self.MOVEMENT_TABLE)

            if flag == 0:
                pass
            elif flag == 1:
                pass
            elif flag == -1:
                pass

            self.update_q_table((old_state_a, Agent.blur(old_state_b)), (state_a, Agent.blur(state_b)), action_for_a, now_reward_a)
            self.update_q_table((old_state_b, Agent.blur(old_state_a)), (state_b, Agent.blur(state_a)), action_for_b, now_reward_b)
            round_cnt += 1
        self.total_round += round_cnt
        self.game_round += 1

    @staticmethod
    def test(Agent1, Agent2):
        ROUND_PER_TEST = Agent1.HYPERPARAMETER_DICT["ROUND_PER_TEST"]
        print(f"Start test after {Agent1.game_round} games.")
        win_cnt = 0
        for i in range(ROUND_PER_TEST):
            while True:
                state_a, state_b = (0, 0), (0, 0)
                action_a = Agent1.choose_action((state_a, state_b), False)  # 按照 Q 表选择动作
                action_b = Agent2.choose_action((state_b, state_a), False)
                flag, state_a, state_b, _, _ = Agent.judge(state_a, state_b, action_a, action_b, Agent1.MOVEMENT_TABLE)
                if flag != 0:
                    if flag == 1:
                        win_cnt += 1
                    break
        print(f"Finish test after {Agent1.game_round} games.")
        print(f"Win rate: {win_cnt / ROUND_PER_TEST:.3%}.")
        Agent1.test_data[0].append(Agent1.game_round)
        Agent1.test_data[1].append(win_cnt / ROUND_PER_TEST)


if __name__ == "__main__":
    random.seed(42)
    try:
        remove("log.txt")
    except FileNotFoundError:
        pass
    Trainee = Agent()
    Trainee.init_q_table_and_configs()
    print(Trainee.HYPERPARAMETER_DICT)
    Randomer = Foolish(Trainee.MOVEMENT_TABLE)
    TOTAL_GAME_ROUND = Trainee.HYPERPARAMETER_DICT["TOTAL_GAME_ROUND"]
    try:
        while True:
            if Trainee.game_round % 50000 == 0:
                Agent.test(Trainee, Randomer)
            Trainee.play_round()
            if Trainee.total_round >= TOTAL_GAME_ROUND:
                break
        print("Finish all games.")
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        exit()
    else:
        Trainee.save_q_table_and_configs()
    finally:
        plt.plot(Trainee.test_data[0], Trainee.test_data[1], color="black", label="Win Rate", linewidth=1, linestyle="-", marker="o")
        plt.title("Win Rate", fontsize=14, fontweight="bold")
        plt.xlabel("Games", fontsize=12)
        plt.ylabel("Win Rate", fontsize=12)
        plt.legend(loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f"win_rate_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}.png", dpi=300)
        plt.show()
