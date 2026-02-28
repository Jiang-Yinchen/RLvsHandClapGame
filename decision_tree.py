import sys
import joblib
from sklearn.tree import DecisionTreeClassifier, export_text
import statistics

def data_prepare(path):
    table = joblib.load(path)
    usable_movements = {}
    for kk in table.keys():
        actions = table[kk]
        actions = [(k, v["reward"]) for k, v in actions.items()]
        rewards = [r for _, r in actions]
        mean = statistics.mean(rewards)
        rewards = [i - mean for i in rewards]
        maximum = max(rewards)
        if maximum == 0:
            rewards = [1.0] * len(rewards)
        else:
            rewards = [i / maximum for i in rewards]
        actions = [(k, nr) for (k, _), nr in zip(actions, rewards)]
        for k, v in actions:
            if k not in usable_movements:
                usable_movements[k] = ([], [])
            def unblur(x):
                return [0, 1, 2, 3, 5, 7, 10, 13][x]
            usable_movements[k][0].append([kk[0][0], kk[0][1], unblur(kk[1][0]), unblur(kk[1][1])])
            usable_movements[k][1].append(int(v >= 0.9))
    return usable_movements

def generated_tree(usable_movements):
    fullname = {
        "生": "生化",
        "防": "防御",
        "飞": "飞天",
        "单": "单刀",
        "双": "双刀",
        "弯": "弯刀",
        "刺": "刺刀",
        "肥": "大肥刀",
        "镖": "回旋镖",
        "零": "零技能",
        "一": "一技能",
        "二": "二技能",
        "三": "三技能",
        "四": "四技能",
        "五": "五技能",
        "胡": "胡萝卜",
        "菜": "菜刀",
        "厨": "厨师",
        "地": "地雷",
        "机": "机器人"
    }
    rules = {}
    for movement, usable in usable_movements.items():
        model = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=42, min_samples_leaf=0.1, class_weight="balanced")
        model.fit(usable[0], usable[1])
        rule = export_text(model, feature_names=["我的生化数量", "我的连续生化数量", "对方生化数量", "对方连续生化数量"], class_names=["可以出" + fullname[movement], "不出" + fullname[movement]])
        rules[movement] = rule
    return rules

def main(arg):
    rules = generated_tree(data_prepare(arg))
    for rule in rules.values():
        print(rule)
        print()



if __name__ == "__main__":
    main(sys.argv[1])
