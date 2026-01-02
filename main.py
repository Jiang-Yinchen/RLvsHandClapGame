from math import inf
import random
import time
from os import remove

ALPHA_0 = 0.2
ALPHA_MIN = 0.02
GAMMA = 0.85
TOTAL_EPISODES = 15
DECAY_EPISODES = 10
TOTAL_GAME_ROUND = 1000000
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
    print(logs, end="")
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
                                Q_table[((i, k), (j, s))].update({t: {"reward": random.random() * 2 - 1, "episode": 0}})
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

    new_episode_max_reward = max([Q_table[new_state][i]["reward"] for i in Q_table[new_state].keys()])  # 新一轮的最大奖励
    Q_table[old_state][action]["reward"] = Q_table[old_state][action]["reward"] + get_alpha(Q_table[old_state][action]["episode"]) * (reward + GAMMA * new_episode_max_reward - Q_table[old_state][action]["reward"])  # 更新 Q 表


def play_round():  # 开始一轮游戏
    global MOVEMENT_TABLE, game_round, total_round, Q_table
    state_a, state_b = (0, 0), (0, 0)
    flag = True
    round_cnt = 0
    while flag:  # 循环直到结束一轮游戏
        # log(f"Start game {game_round} round {round_cnt}.")
        old_state_a, old_state_b = (state_a, state_b), (state_b, state_a)
        movements_for_a, movements_for_b = Q_table[(state_a, state_b)], Q_table[(state_b, state_a)]
        max_reward_for_a, max_reward_for_b = -inf, -inf
        now_reward_a, now_reward_b = 0, 0
        action_for_a, action_for_b = None, None
        if random.randrange(10) != 0:
            for i in movements_for_a.keys():
                i_reward = movements_for_a[i]["reward"]
                if i_reward > max_reward_for_a:
                    max_reward_for_a = i_reward
                    action_for_a = i
        else:
            action_for_a = random.choice(list(movements_for_a.keys()))
        if random.randrange(10) != 0:
            for i in movements_for_b.keys():
                i_reward = movements_for_b[i]["reward"]
                if i_reward > max_reward_for_b:
                    max_reward_for_b = i_reward
                    action_for_b = i
        else:
            action_for_b = random.choice(list(movements_for_b.keys()))
        if action_for_a is None or action_for_b is None:
            log("What the fuck!")
            raise KeyError
        movements_for_a[action_for_a]["episode"] += 1
        movements_for_b[action_for_b]["episode"] += 1

        if action_for_b in MOVEMENT_TABLE[action_for_a]["kill"]:
            log(f"In game {game_round}, Player A uses {action_for_a} kills {action_for_b}!")
            flag = False
            now_reward_a += 5
            now_reward_b -= 5
        elif action_for_a in MOVEMENT_TABLE[action_for_b]["kill"]:
            log(f"In game {game_round}, Player B uses {action_for_b} kills {action_for_a}!")
            flag = False
            now_reward_a -= 5
            now_reward_b += 5
        else:
            if MOVEMENT_TABLE[action_for_a]["combo"] != 0:
                state_a = (state_a[0], 0)
            elif MOVEMENT_TABLE[action_for_a]["need"] < 0:
                state_a = (state_a[0], 0)
            elif MOVEMENT_TABLE[action_for_a]["need"] != 0:
                state_a = (state_a[0], min(state_a[1] + 1, 5))
            state_a = (min(state_a[0] + MOVEMENT_TABLE[action_for_a]["need"], 10), state_a[1])
            if MOVEMENT_TABLE[action_for_b]["combo"] != 0:
                state_b = (state_b[0], 0)
            elif MOVEMENT_TABLE[action_for_b]["need"] < 0:
                state_b = (state_b[0], 0)
            elif MOVEMENT_TABLE[action_for_b]["need"] != 0:
                state_b = (state_b[0], min(state_b[1] + 1, 5))
            state_b = (min(state_b[0] + MOVEMENT_TABLE[action_for_b]["need"], 10), state_b[1])
            # log(f"In game {game_round}, Player A uses {action_for_a} and Player B uses {action_for_b}!")
            now_reward_a += 2 ** -game_round
            now_reward_b += 2 ** -game_round

        # log(f"Now state is {state_a} and {state_b} in game {game_round}.")
        update_q_table(old_state_a, (state_a, state_b), action_for_a, now_reward_a)
        update_q_table(old_state_b, (state_b, state_a), action_for_b, now_reward_b)
        round_cnt += 1
    log(f"Finish game {game_round} after {round_cnt} round(s).")
    total_round += round_cnt
    log(f"Total round: {total_round}.")
    game_round += 1


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
            play_round()
            if total_round >= TOTAL_GAME_ROUND:
                break
        log("Finish all games.")
    except KeyboardInterrupt:
        log("KeyboardInterrupt")
    except KeyError as e:
        log(f"KeyError: {e}")
    except Exception as e:
        log(f"Exception: {e}")
    finally:
        end_log()
