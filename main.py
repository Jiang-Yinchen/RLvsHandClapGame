import sys
import train
import decision_tree

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "-t":
        train.main(sys.argv[2:])
    elif mode == "-d":
        decision_tree.main(sys.argv[2:])
    else:
        pass