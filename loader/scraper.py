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
YEARS = [2005]
WEEKS = [i for i in range(1, 18)]
# WEEKS = [i for i in range(1, 3)]
PFF_LOGGED_IN = False


def get_url(base_url: str, year: int, week: int, team: str, unit: str):
    return f"{base_url}{year}/{week}/{team}/{unit}/"


TEAMS = {
    "ARI": {"full_dash": "arizona-cardinals", "mascot": "cardinals", "abbreviation": "ARI"},
    "ATL": {"full_dash": "atlanta-falcons", "mascot": "falcons", "abbreviation": "ATL"},
    "BAL": {"full_dash": "baltimore-ravens", "mascot": "ravens", "abbreviation": "BAL"},
    "BUF": {"full_dash": "buffalo-bills", "mascot": "bills", "abbreviation": "BUF"},
    "CAR": {"full_dash": "carolina-panthers", "mascot": "panthers", "abbreviation": "CAR"},
    "CHI": {"full_dash": "chicago-bears", "mascot": "bears", "abbreviation": "CHI"},
    "CIN": {"full_dash": "cincinnati-bengals", "mascot": "bengals", "abbreviation": "CIN"},
    "CLE": {"full_dash": "cleveland-browns", "mascot": "browns", "abbreviation": "CLE"},
    "DAL": {"full_dash": "dallas-cowboys", "mascot": "cowboys", "abbreviation": "DAL"},
    "DEN": {"full_dash": "denver-broncos", "mascot": "broncos", "abbreviation": "DEN"},
    "DET": {"full_dash": "detroit-lions", "mascot": "lions", "abbreviation": "DET"},
    "GB": {"full_dash": "green-bay-packers", "mascot": "packers", "abbreviation": "GB"},
    "HOU": {"full_dash": "houston-texans", "mascot": "texans", "abbreviation": "HOU"},
    "IND": {"full_dash": "indianapolis-colts", "mascot": "colts", "abbreviation": "IND"},
    "JAC": {"full_dash": "jacksonville-jaguars", "mascot": "jaguars", "abbreviation": "JAC"},
    "KC": {"full_dash": "kansas-city-chiefs", "mascot": "chiefs", "abbreviation": "KC"},
    # "LV": {"full_dash": "las-vegas-raiders", "mascot": "raiders", "abbreviation": "LV"},
    "OAK": {"full_dash": "oakland-raiders", "mascot": "raiders", "abbreviation": "OAK"},
    # "LAC": {"full_dash": "los-angeles-chargers", "mascot": "chargers", "abbreviation": "LAC"},
    "SD": {"full_dash": "san-diego-chargers", "mascot": "chargers", "abbreviation": "SD"},
    # "LAR": {"full_dash": "los-angeles-rams", "mascot": "rams", "abbreviation": "LAR"},
    "STL": {"full_dash": "st-louis-rams", "mascot": "rams", "abbreviation": "STL"},
    "MIA": {"full_dash": "miami-dolphins", "mascot": "dolphins", "abbreviation": "MIA"},
    "MIN": {"full_dash": "minnesota-vikings", "mascot": "vikings", "abbreviation": "MIN"},
    "NE": {"full_dash": "new-england-patriots", "mascot": "patriots", "abbreviation": "NE"},
    "NO": {"full_dash": "new-orleans-saints", "mascot": "saints", "abbreviation": "NO"},
    "NYG": {"full_dash": "new-york-giants", "mascot": "giants", "abbreviation": "NYG"},
    "NYJ": {"full_dash": "new-york-jets", "mascot": "jets", "abbreviation": "NYJ"},
    "PHI": {"full_dash": "philadelphia-eagles", "mascot": "eagles", "abbreviation": "PHI"},
    "PIT": {"full_dash": "pittsburgh-steelers", "mascot": "steelers", "abbreviation": "PIT"},
    "SF": {"full_dash": "san-francisco-49ers", "mascot": "49ers", "abbreviation": "SF"},
    "SEA": {"full_dash": "seattle-seahawks", "mascot": "seahawks", "abbreviation": "SEA"},
    "TB": {"full_dash": "tampa-bay-buccaneers", "mascot": "buccaneers", "abbreviation": "TB"},
    "TEN": {"full_dash": "tennessee-titans", "mascot": "titans", "abbreviation": "TEN"},
    # "WAS": {"full_dash": "washington-commanders", "mascot": "commanders", "abbreviation": "WAS"},
    # "WAS": {"full_dash": "washington-football-team", "mascot": "washington", "abbreviation": "WAS"},
    "WAS": {"full_dash": "washington-redskins", "mascot": "redskins", "abbreviation": "WAS"},
}


class Unit(Enum):
    OFF = "offense"
    DEF = "defense"
    ST = "special-teams"


class TeamGameData:
    def __init__(self, team: dict, year: int, week: int):
        self.week = week
        self.year = year
        self.team = team
        self.offense = self._create_offense()
        self.defense = self._create_defense()
        self.special_teams = self._create_special_teams()

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

    @staticmethod
    def get_headers():
        headers = ["YEAR", "WEEK", "TEAM"]
        offense = [key for key in TeamGameData._create_offense()]
        defense = [key for key in TeamGameData._create_defense()]
        st = [key for key in TeamGameData._create_special_teams()]
        headers += offense + defense + st

        return headers

    def get_row(self):
        row = [self.year, self.week, self.team["abbreviation"]]
        for value in self.offense.values():
            row.append(value[0])
        for value in self.defense.values():
            row.append(value[0])
        for value in self.special_teams.values():
            row.append(value[0])

        return row

    def __str__(self):
        return f"{self.team['full_dash']}"


def login_pff(browser: WebDriver):
    global PFF_LOGGED_IN
    if PFF_LOGGED_IN:
        return

    auth_url = "https://auth.pff.com/"
    browser.get(auth_url)
    time.sleep(3)  # give it time to load
    username = os.getenv("PFF_USER")
    password = os.getenv("PFF_PASS")

    browser.find_element(value="login-form_email").send_keys(username)
    browser.find_element(value="login-form_password").send_keys(password)
    browser.find_element(value="sign-in").click()
    time.sleep(3)

    browser.get("https://premium.pff.com/")
    time.sleep(3)

    browser.find_element(by=by.By.CLASS_NAME, value="g-btn--green").click()
    time.sleep(3)

    PFF_LOGGED_IN = True


def scrape_pff(browser: WebDriver, filename: str):
    assert len(TEAMS) == 32
    login_pff(browser)

    base_url = BASE_URL

    headers = TeamGameData.get_headers()
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)

        for year in YEARS:
            for week in WEEKS:
                for team in TEAMS:
                    team_game_data = TeamGameData(team=TEAMS[team], year=year, week=week)
                    for unit in Unit:
                        url = get_url(
                            base_url=base_url,
                            year=year,
                            week=week,
                            team=str(team_game_data.team["full_dash"]),
                            unit=str(unit.value),
                        )
                        # print(f"{url =}")
                        browser.get(url)
                        time.sleep(3)  # give it time to load
                        html = browser.page_source
                        soup = BeautifulSoup(html, "html.parser")
                        all_grades_raw = soup.find_all(class_="kyber-table-body__scrolling-rows-container")
                        if not all_grades_raw:
                            continue  # no team data, probably on bye
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

                    row = team_game_data.get_row()
                    csvwriter.writerow(row)

                    # if team_game_data.team == "CLE" or team_game_data.team == "cleveland-browns":
                    #     return


class GameResultsData:
    def __init__(self, year: int, week: int):
        self.week = week
        self.year = year
        self.home_team = ""
        self.away_team = ""
        self.home_score = -1
        self.away_score = -1

    def get_row(self):
        row = [self.year, self.week, self.home_team, self.away_team, self.home_score, self.away_score]

        return row

    @staticmethod
    def get_headers():
        headers = ["YEAR", "WEEK", "HOME_TEAM", "AWAY_TEAM", "HOME_SCORE", "AWAY_SCORE"]

        return headers


def scrape_espn_results(browser: WebDriver, filename: str):
    base_url = f"https://www.espn.com/nfl/scoreboard/_/"
    assert len(TEAMS) == 32
    teams = {TEAMS[abbr]["mascot"]: abbr for abbr in TEAMS}

    headers = GameResultsData.get_headers()
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)

        for year in YEARS:
            for week in WEEKS:
                url = f"{base_url}week/{week}/year/{year}/"
                browser.get(url)
                time.sleep(2)  # give it time to load
                html = browser.page_source
                soup = BeautifulSoup(html, "html.parser")
                all_game_results = soup.find_all(class_="ScoreboardScoreCell__Competitors")
                if not all_game_results:
                    continue  # messed up
                for game_result in all_game_results:
                    result = GameResultsData(year=year, week=week)
                    for idx, team in enumerate(game_result):  # team 0 is away, team 1 is home
                        if idx == 0:
                            result.away_team, result.away_score = _get_team_and_score(team, teams)
                        elif idx == 1:
                            result.home_team, result.home_score = _get_team_and_score(team, teams)

                    if result.away_score != -1 and result.home_score != -1:
                        row = result.get_row()
                        csvwriter.writerow(row)


def _get_team_and_score(team, teams: dict) -> (str, int):
    team_name = team.find_all(class_="ScoreCell__TeamName")[0].string
    team_abbr = teams[str(team_name).lower()]
    if len(team.find_all(class_="ScoreCell__Score")) == 0:
        return team_abbr, -1
    score = int(team.find_all(class_="ScoreCell__Score")[0].string)

    return team_abbr, score


def main():
    browser = webdriver.Chrome(r"C:\WebDriver_\chromedriver.exe")

    scrape_pff(browser, "../data/team_grades.csv")
    scrape_espn_results(browser, "../data/results.csv")

    browser.close()


if __name__ == '__main__':
    main()
