from math import exp, inf
import matplotlib.pyplot as plt
import random
import time
from os import remove

ALPHA_0 = 0.2
ALPHA_MIN = 0.02
GAMMA = 0.85
TOTAL_EPISODES = 15
DECAY_EPISODES = 10
EPSILON_START = 0.2
EPSILON_END = 0.01
EPSILON_DECAY = 1e-7
TOTAL_GAME_ROUND = 5000000
ROUND_PER_TEST = 2000
MOVEMENT_TABLE = {
    "生": {"kill": {"零"}, "need": 1, "combo": 0},
    "防": {"kill": set(), "need": 0, "combo": 0},
    "飞": {"kill": set(), "need": 0, "combo": 0},
    "单": {"kill": {"生", "地"}, "need": -1, "combo": 0},
    "双": {"kill": {"生", "一", "地"}, "need": -2, "combo": 0},
    "弯": {"kill": {"单", "双", "镖", "地", "机"}, "need": -1, "combo": 0},
    "刺": {"kill": {"弯", "肥", "地"}, "need": -1, "combo": 0},
    "肥": {"kill": {"防", "飞", "弯"}, "need": -5, "combo": 0},
    "镖": {"kill": {"飞"}, "need": -3, "combo": 0},
    "零": {"kill": {"防"}, "need": 0, "combo": 0},
    "一": {"kill": set(), "need": 1, "combo": 1},
    "二": {"kill": set(), "need": 2, "combo": 2},
    "三": {"kill": set(), "need": 3, "combo": 3},
    "四": {"kill": set(), "need": 4, "combo": 4},
    "五": {"kill": set(), "need": 5, "combo": 5},
    "胡": {"kill": {"生", "厨"}, "need": -1, "combo": 0},
    "菜": {"kill": {"胡"}, "need": -1, "combo": 0},
    "厨": {"kill": {"菜"}, "need": -2, "combo": 0},
    "地": {"kill": {"生", "一", "二", "三", "四", "五"}, "need": -3, "combo": 0},
    "机": {"kill": {"生", "防", "飞", "单", "双", "刺", "肥", "镖", "一", "二", "三", "四", "五", "胡", "菜", "厨", "地"}, "need": -10, "combo": 0}
}
START_TIME = -inf
Q_table = {}
game_round = 0  # 总游戏数
total_round = 0  # 总回合数
logs = ""
test_data = [[], []]


def get_time():  # 获取运行时间
    global START_TIME
    return time.time() - START_TIME


def log(message):  # 记录日志
    global logs
    logs += f"{get_time():.3f}: {message}\n"
    if len(logs) > 1000000:
        end_log()


def end_log():  # 结束一轮日志，并写入文件
    global logs
    # print(logs, end="")
    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(logs)
    logs = ""


def init_q_table():  # 初始化 Q 表
    global MOVEMENT_TABLE
    log("Start initializing Q_table.")
    cnt_movement = 0
    cnt_state = 0
    # 使用循环变量来纪念 Dijkstra
    for i in range(11):  # 我的”生“数量
        for j in range(11):  # 对方的”生“数量
            for k in range(6):  # 我的连续”生“数量
                for s in range(6):  # 对方的连续”生“数量
                    if i >= k and j >= s:
                        Q_table[((i, k), (j, s))] = {}
                        for t in MOVEMENT_TABLE.keys():
                            if -MOVEMENT_TABLE[t]["need"] <= i and MOVEMENT_TABLE[t]["combo"] <= k:
                                Q_table[((i, k), (j, s))].update({t: {"reward": 0, "episode": 0}})
                                cnt_movement += 1
                        cnt_state += 1
    log(f"Q_table {hex(hash(str(Q_table)))} has been initialized with {cnt_state} state(s) and {cnt_movement} movement(s).")


def update_q_table(old_state, new_state, action, reward):  # 更新 Q 表
    global Q_table, GAMMA

    def get_alpha(episode):  # 获取学习率
        global ALPHA_0, ALPHA_MIN, DECAY_EPISODES
        if episode < DECAY_EPISODES:
            return ALPHA_0 - (ALPHA_0 - ALPHA_MIN) * episode / DECAY_EPISODES
        return ALPHA_MIN

    Q_table[old_state][action]["episode"] += 1
    new_episode_max_reward = max([Q_table[new_state][i]["reward"] for i in Q_table[new_state].keys()])  # 新一轮的最大奖励
    Q_table[old_state][action]["reward"] = Q_table[old_state][action]["reward"] + get_alpha(Q_table[old_state][action]["episode"]) * (reward + GAMMA * new_episode_max_reward - Q_table[old_state][action]["reward"])  # 更新 Q 表
    Q_table[old_state][action]["reward"] = max(min(Q_table[old_state][action]["reward"], 10), -10)  # 限制奖励范围


def choose_action(state):  # 选择动作
    def get_epsilon(rounds):  # 获取探索率
        global EPSILON_START, EPSILON_END, EPSILON_DECAY
        return EPSILON_END + (EPSILON_START - EPSILON_END) * exp(-1 * rounds * EPSILON_DECAY)

    movements = Q_table[state]
    max_reward = -inf
    action = None
    if random.random() >= get_epsilon(total_round):
        for i in movements.keys():
            i_reward = movements[i]["reward"]
            if i_reward >= max_reward:
                max_reward = i_reward
                action = i
    else:
        action = random.choice(list(movements.keys()))
    return action


def judge(player_a_state, player_b_state, player_a_action, player_b_action):
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
        elif MOVEMENT_TABLE[player_a_action]["need"] < 0:
            player_a_state = (player_a_state[0], 0)
        elif MOVEMENT_TABLE[player_a_action]["need"] != 0:
            player_a_state = (player_a_state[0], min(player_a_state[1] + 1, 5))
        player_a_state = (min(player_a_state[0] + MOVEMENT_TABLE[player_a_action]["need"], 10), player_a_state[1])
        if MOVEMENT_TABLE[player_b_action]["combo"] != 0:
            player_b_state = (player_b_state[0], 0)
        elif MOVEMENT_TABLE[player_b_action]["need"] < 0:
            player_b_state = (player_b_state[0], 0)
        elif MOVEMENT_TABLE[player_b_action]["need"] != 0:
            player_b_state = (player_b_state[0], min(player_b_state[1] + 1, 5))
        player_b_state = (min(player_b_state[0] + MOVEMENT_TABLE[player_b_action]["need"], 10), player_b_state[1])
        now_reward_a += 2 ** -game_round
        now_reward_b += 2 ** -game_round
    return ended, player_a_state, player_b_state, now_reward_a, now_reward_b


def play_round():  # 开始一轮游戏
    global MOVEMENT_TABLE, game_round, total_round, Q_table
    state_a, state_b = (0, 0), (0, 0)
    flag = 0
    round_cnt = 0
    while flag == 0:  # 循环直到结束一轮游戏
        # log(f"Start game {game_round} round {round_cnt}.")
        old_state_a, old_state_b = state_a, state_b

        action_for_a, action_for_b = choose_action((state_a, state_b)), choose_action((state_b, state_a))
        flag, state_a, state_b, now_reward_a, now_reward_b = judge(state_a, state_b, action_for_a, action_for_b)

        if flag == 0:
            pass
            # log(f"In game {game_round}, Player A uses {player_a_action} and Player B uses {player_b_action}!")
        elif flag == 1:
            log(f"In game {game_round}, Player A uses {action_for_a} kills {action_for_b}!")
        elif flag == -1:
            log(f"In game {game_round}, Player B uses {action_for_b} kills {action_for_a}!")

        # log(f"Now state is {state_a} and {state_b} in game {game_round}.")
        update_q_table((old_state_a, old_state_b), (state_a, state_b), action_for_a, now_reward_a)
        update_q_table((old_state_b, old_state_a), (state_b, state_a), action_for_b, now_reward_b)
        round_cnt += 1
    log(f"Finish game {game_round} after {round_cnt} round(s).")
    total_round += round_cnt
    log(f"Total round: {total_round}.")
    game_round += 1


def test():
    global ROUND_PER_TEST, game_round, Q_table, test_data
    log(f"Start test after {game_round} games.")
    win_cnt = 0
    for i in range(ROUND_PER_TEST):
        while True:
            state_a, state_b = (0, 0), (0, 0)
            action_a = choose_action((state_a, state_b))  # 按照 Q 表选择动作
            action_b = random.choice(list(Q_table[(state_b, state_a)].keys()))  # 随机选择动作
            flag, state_a, state_b, _, _ = judge(state_a, state_b, action_a, action_b)
            if flag != 0:
                if flag == 1:
                    win_cnt += 1
                break
    log(f"Finish test after {game_round} games.")
    log(f"Win rate: {win_cnt / ROUND_PER_TEST:.2%}.")
    print(f"Win rate: {win_cnt / ROUND_PER_TEST:.2%}")
    test_data[0].append(game_round)
    test_data[1].append(win_cnt / ROUND_PER_TEST)


if __name__ == '__main__':
    random.seed(42)
    START_TIME = time.time()
    try:
        remove("log.txt")
    except FileNotFoundError:
        pass
    log("Start game.")
    try:
        init_q_table()
        while True:
            if game_round % 25000 == 0:
                test()
            play_round()
            if total_round >= TOTAL_GAME_ROUND:
                break
        log("Finish all games.")
    except KeyboardInterrupt:
        log("KeyboardInterrupt")
    finally:
        end_log()
        plt.plot(test_data[0], test_data[1], color="black", label="Win Rate", linewidth=1, linestyle="-", marker="o")
        plt.title('Win Rate', fontsize=14, fontweight='bold')
        plt.xlabel('Games', fontsize=12)
        plt.ylabel('Win Rate', fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig('win_rate.png', dpi=300)
        plt.show()
