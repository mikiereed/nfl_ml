import csv
import os
from selenium import webdriver
from selenium.webdriver.common import by
from selenium.webdriver.chrome.webdriver import WebDriver
from bs4 import BeautifulSoup
import time
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


BASE_URL = "https://premium.pff.com/nfl/games/"
YEARS = [2022]
WEEKS = [5]
PFF_LOGGED_IN = False


def get_url(base_url: str, year: int, week: int, team: str, unit: str):
    return f"{base_url}{year}/{week}/{team}/{unit}/"


class Team(Enum):
    CLE = "cleveland-browns"
    DAL = "dallas-cowboys"
    LAR = "los-angeles-rams"
    SF = "san-francisco-49ers"


class Unit(Enum):
    OFF = "offense"
    DEF = "defense"
    ST = "special-teams"


class TeamGameData:
    def __init__(self, team: Team):
        self.team = team.name
        self.offense = self._create_offense()
        self.defense = self._create_defense()
        self.special_teams = self._create_special_teams()
        self.headers = None

    @staticmethod
    def _create_offense() -> dict:
        return {
            "LG": [0.0, 0],  # [grade, plays]
            "LT": [0.0, 0],
            "C": [0.0, 0],
            "RT": [0.0, 0],
            "RG": [0.0, 0],
            "TE": [0.0, 0],
            "HB": [0.0, 0],
            "QB": [0.0, 0],
            "WR": [0.0, 0],
        }

    @staticmethod
    def _create_defense():
        return {
            "DE": [0.0, 0],
            "DT": [0.0, 0],
            "LB": [0.0, 0],
            "CB": [0.0, 0],
            "S": [0.0, 0],
        }

    @staticmethod
    def _create_special_teams():
        return {
            "ST": [0.0, 0],
            "LS": [0.0, 0],
            "K": [0.0, 0],
            "P": [0.0, 0],
        }

    def update_position_grade(self, unit: str, position: str, grade: float, plays: int) -> None:
        if unit == "offense":
            self._update_position_grade(self.offense, position, grade, plays)
        elif unit == "defense":
            self._update_position_grade(self.defense, position, grade, plays)
        elif unit == "special_teams":
            self._update_position_grade(self.special_teams, position, grade, plays)

    @staticmethod
    def _update_position_grade(unit: dict, position: str, grade: float, plays: int) -> None:
        previous_grade = unit[position]
        new_grade = ((previous_grade[0] * previous_grade[1]) + (grade * plays)) / (previous_grade[1] + plays)
        unit[position] = [new_grade, previous_grade[1] + plays]

    def get_headers(self):
        headers = ["TEAM"]
        offense = [key for key in self.offense]
        defense = [key for key in self.defense]
        st = [key for key in self.special_teams]
        headers += offense + defense + st

        return headers

    def get_rows(self):
        rows = [self.team]
        for value in self.offense.values():
            rows.append(value[0])
        for value in self.defense.values():
            rows.append(value[0])
        for value in self.special_teams.values():
            rows.append(value[0])

        return rows

    def __str__(self):
        return f"{self.team}"


def login_pff(browser: WebDriver):
    global PFF_LOGGED_IN
    if PFF_LOGGED_IN:
        return

    auth_url = "https://auth.pff.com/"
    browser.get(auth_url)
    time.sleep(2)  # give it time to load
    username = os.getenv("PFF_USER")
    password = os.getenv("PFF_PASS")

    browser.find_element(value="login-form_email").send_keys(username)
    browser.find_element(value="login-form_password").send_keys(password)
    browser.find_element(value="sign-in").click()
    time.sleep(2)

    browser.get("https://premium.pff.com/")
    time.sleep(2)

    browser.find_element(by=by.By.CLASS_NAME, value="g-btn--green").click()
    time.sleep(2)

    PFF_LOGGED_IN = True


def scrape_pff(browser: WebDriver, filename: str):
    login_pff(browser)

    base_url = BASE_URL

    headers = TeamGameData(Team.LAR).get_headers()
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)

        for year in YEARS:
            for week in WEEKS:
                for team in Team:
                    team_game_data = TeamGameData(team)
                    for unit in Unit:
                        url = get_url(
                            base_url=base_url,
                            year=year,
                            week=week,
                            team=str(team.value),
                            unit=str(unit.value),
                        )
                        # print(f"{url =}")
                        browser.get(url)
                        time.sleep(2)  # give it time to load
                        html = browser.page_source
                        soup = BeautifulSoup(html, "html.parser")
                        all_grades_raw = soup.find_all(class_="kyber-table-body__scrolling-rows-container")
                        rows_raw = all_grades_raw[0].find_all(class_="kyber-table-body__row")
                        for row in rows_raw:
                            position_and_plays = row.find_all(class_="kyber-table-body-cell")
                            position = position_and_plays[1].string
                            if not position:
                                continue
                            plays = int(position_and_plays[2].string)
                            grade = float(row.find_all(class_="kyber-grade-badge__info-text")[0].string)

                            # offense
                            if unit.value == "offense":
                                if position.startswith("TE"):
                                    position = "TE"
                                elif position.endswith("WR"):
                                    position = "WR"
                                elif position.endswith("FB"):
                                    position = "HB"
                                position_grade = team_game_data.offense.get(position, "NOT VALID")
                                if position_grade == "NOT VALID":
                                    raise "Yo Yo Yo, this is not cool!!!  e o"
                                if position_grade[1]:
                                    team_game_data.update_position_grade(
                                                                         unit="offense",
                                                                         position=position,
                                                                         grade=grade,
                                                                         plays=plays,
                                                                        )
                                else:
                                    team_game_data.offense[position] = [grade, plays]

                            # defense
                            if unit.value == "defense":
                                if position.startswith("D"):
                                    position = position.replace("D", "")

                                if position.endswith("LB"):
                                    position = "LB"
                                elif position.endswith("T"):
                                    position = "DT"
                                elif position.endswith("E"):
                                    position = "DE"
                                elif position == "FS" or position == "SS":
                                    position = "S"
                                elif position.endswith("CB"):
                                    position = "CB"

                                position_grade = team_game_data.defense.get(position, None)
                                if position_grade is None:
                                    raise "Yo Yo Yo, this is not cool!!!  e o"
                                if position_grade[1]:
                                    team_game_data.update_position_grade(
                                                                         unit="defense",
                                                                         position=position,
                                                                         grade=grade,
                                                                         plays=plays,
                                                                        )
                                else:
                                    team_game_data.defense[position] = [grade, plays]

                            # special teams
                            if unit.value == "special-teams":
                                position_grade = team_game_data.special_teams.get(position, None)
                                if position_grade is None:
                                    raise "Yo Yo Yo, this is not cool!!!  e o"
                                if position_grade[1]:
                                    team_game_data.update_position_grade(
                                        unit="special_teams",
                                        position=position,
                                        grade=grade,
                                        plays=plays,
                                    )
                                else:
                                    team_game_data.special_teams[position] = [grade, plays]

                    # week
                    # year
                    rows = team_game_data.get_rows()
                    csvwriter.writerow(rows)


def main():
    browser = webdriver.Chrome(r"C:\WebDriver_\chromedriver.exe")

    scrape_pff(browser, "team_grades.csv")

    browser.close()


if __name__ == '__main__':
    main()
