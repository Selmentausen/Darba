from flask import Flask, jsonify

app = Flask(__name__)


@app.route('/bones')
def get_bones():
    data = dict()
    data['right_hand'] = {'joint_target': (0, 0, 0), 'effector': (-0.5, 0.25, 0)}
    data['left_hand'] = {'joint_target': (0, 0, 0), 'effector': (0, 0, 0)}
    data['root'] = (0.3, 0.3, 0.3)
    return jsonify(data)


def main():
    app.run(host='127.0.0.1', port='8080')



def test():
    l_hand = (1, 1, 0)
    r_hand = (2, 1, 2)
    n = abs(((l_hand[0] - r_hand[0]) ** 2 + (l_hand[1] - r_hand[1]) ** 2 + (l_hand[2] - r_hand[2]) ** 2) ** 0.5)
    print(n)


if __name__ == '__main__':
    test()
    # main()
