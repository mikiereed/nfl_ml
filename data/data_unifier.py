import csv
import os


def unify_data() -> None:
    # create dictionary of team grades
    team_grades = dict()
    x = []
    y = []
    team_grades_dir = "./team_grades/"
    for team_grades_file in os.listdir(team_grades_dir):
    # for i in range(6, 22):
        with open(f"{team_grades_dir}{team_grades_file}", "r") as f:
            lines = f.readlines()
            for idx, team_grade in enumerate(lines):
                if idx == 0:
                    continue  # headers
                team_grade = team_grade.replace("\n", "").split(",")
                team_grades[(int(team_grade[0]), int(team_grade[1]), team_grade[2])] = team_grade[3:]

        # combine teams from results
    results_dir = "./results/"
    for results_file in os.listdir(results_dir):
        with open(f"{results_dir}{results_file}", "r") as f:
            lines = f.readlines()
            for idx, result in enumerate(lines):
                if idx == 0:
                    continue  # headers
                result = result.replace("\n", "").split(",")
                home_x = team_grades[(int(result[0]), int(result[1]), result[2])]
                away_x = team_grades[(int(result[0]), int(result[1]), result[3])]
                x.append(home_x + away_x)
                home_win = 1 if int(result[4]) > int(result[5]) else 0
                y.append(home_win)

    with open("data.csv", "w", newline='') as f:
        features = len(x[0])
        headers = [f"x_{i}" for i in range(features)]
        headers.append("y")
        csvwriter = csv.writer(f)
        csvwriter.writerow(headers)
        for row in range(len(x)):
            x[row].append(y[row])
            csvwriter.writerow(x[row])


if __name__ == "__main__":
    unify_data()
