import sys
import joblib

if __name__ == "__main__":
    s = joblib.load(sys.argv[1])
    while True:
        k = list(map(int, input().split()))
        if k == "":
            break
        print(s[((k[0], k[1]), (k[2], k[3]))])