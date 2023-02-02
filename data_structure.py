

def OddsPortalStructure():
    """
    This function restructures the data pulled from OddsPortal.com. The
    input dataset is divided into 8 different Excel sheets. This is
    restructured into the desired columns. The output is an Excel file
    containing these.

    Requirements:
        - pandas
        - numpy
        - datetime
        - calendar

    """

    # imports
    import pandas as pd
    import numpy as np
    import datetime
    import calendar

    # creating destination dataframe with the desired structure
    df = pd.DataFrame(columns=['date',
                               'home_team',
                               'away_team',
                               'home_score',
                               'away_score',
                               'OddsPortal_1',
                               'OddsPortal_X',
                               'OddsPortal_2',
                               'n_bookmakers'])

    for page in range(1, 9):  # current data consists of 8 separate sheets

        # importing data, no header as this contains data of the date
        df_page = pd.read_excel('output//oddsportal_raw.xlsx', header=None, sheet_name=f'page{page}')

        # renaming columns
        df_page.columns = ['time', 'teams', 'score', '1', 'X', '2', 'n_bookmakers']

        # dropping empty rows
        df_page = df_page.dropna(axis=0, how='all').reset_index(drop=True)

        # creating list for game dates
        dates = [np.nan] * len(df_page)

        # filling out dates
        for idx, val in enumerate(df_page['time']):
            # strings in cells with correct kickoff times contains
            # have a length of 5. So, if the length is higher
            # than this, it is a date (e.g. '22 May 2022')

            if len(val) > 5:
                # this is a date, so the games after this and until
                # new date is encountered are played on this day
                date = val
                dates[idx] = date

        # inserting in dataframe and forward filling
        df_page['date'] = dates
        df_page['date'] = df_page['date'].ffill(axis=0)

        # now the rows used for filling out dates are no longer
        # relevant so these are dropped
        for idx, val in enumerate(df_page['time']):
            if len(val) > 5:
                df_page.drop(idx, inplace=True)

        """
        Now creating new columns to match the desired data structure
        """

        # creating datetime module for exact game time and date
        years = [i[-4:] for i in df_page['date']]
        months = [i[3:6] for i in df_page['date']]
        days = [i[:2] for i in df_page['date']]
        hours = [i[:2] for i in df_page['time']]
        minutes = [i[-2:] for i in df_page['time']]

        # converting months (currently abbreviations) to integers
        months_dict = dict((month, index) for index, month in enumerate(calendar.month_abbr) if month)
        months = [months_dict.get(m) for m in months]

        dates = []
        for y, mo, d, h, mi in zip(years, months, days, hours, minutes):
            dates.append(datetime.datetime(year=int(y),
                                           month=int(mo),
                                           day=int(d),
                                           hour=int(h),
                                           minute=int(mi)))

        # creating team columns
        home_teams = []
        away_teams = []

        for teams in df_page['teams']:
            home_teams.append(teams.split('-')[0].strip())
            away_teams.append(teams.split('-')[1].strip())

        # creating score columns
        home_scores = []
        away_scores = []

        for score in df_page['score']:
            home_scores.append(score.split(':')[0].strip())
            away_scores.append(score.split(':')[1].strip())

        # replacing columns in dataframe
        data = {'date': dates,
                'home_team': home_teams,
                'away_team': away_teams,
                'home_score': home_scores,
                'away_score': away_scores,
                'OddsPortal_1': df_page['1'],
                'OddsPortal_X': df_page['X'],
                'OddsPortal_2': df_page['2'],
                'n_bookmakers': df_page['n_bookmakers']}

        df_page = pd.DataFrame(data)

        # appending to dataframe to contain all games
        df = df.append(df_page, ignore_index=True)

    # sorting games chronologically
    df = df.sort_values(by='date').reset_index(drop=True)

    # creating game id variable
    df.insert(0, 'game_id', '')
    df['game_id'] = [idx + 1 for idx, val in enumerate(df['date'])]

    # changing betting lines for draws as these are stored with dots
    # and not commas
    df['OddsPortal_X'] = [float(str(i).replace('.', '.')) for i in df['OddsPortal_X']]

    # calculating game outcomes and inserting in dataframe
    outcomes = []
    for idx, homescore in enumerate(df['home_score']):
        # if home teams wins
        if homescore > df['away_score'][idx]:
            outcomes.append('1')
        # if away team wins
        if homescore < df['away_score'][idx]:
            outcomes.append('2')
        # if draw
        if homescore == df['away_score'][idx]:
            outcomes.append('X')

    df.insert(6, 'outcome', '')
    df['outcome'] = outcomes

    # exporting as .xlsx file
    df.to_excel('output//oddsportal_structured.xlsx', sheet_name='Data', index=False)

    pass


def RedditQueries():
    """
    This function takes the dataset structured for the Reddit-API requests, and creates a variable containing
    the search queries to be used for gathering the data. This is done for each team.

    Requirements
        - pandas
    """

    # imports and libraries
    import pandas as pd

    # importing structured dataset for reddit
    df_data = pd.read_excel('output//Reddit_structured.xlsx', sheet_name=None)

    # storing sheet names (teams) in list
    sheets = list(df_data.keys())

    # importing dataset with queries
    queries_path = 'QUERIES_PATH'
    df_queries = pd.read_excel(queries_path, sheet_name='queries')

    # checking if all teams are in queries dataset
    for sheet in sheets:
        if sheet not in df_queries['team (OddsPortal)'].tolist():
            print(sheet)

    # setting path to destination excel file
    destination_path = 'output//Reddit_structured_searchquery.xlsx'

    # setting up writer
    writer = pd.ExcelWriter(destination_path, engine='xlsxwriter')

    # looping through each sheet (team) and creating a variable containing the search query. the format is
    # corresponding to the one used in the api requests.
    for idx, sheet in enumerate(sheets):
        # loading dataframe containing given team
        df_team = df_data[sheet]

        # accessing index of query and storing as list
        query_idx = df_queries.index[df_queries['team (OddsPortal)'] == sheet]
        queries = df_queries['queries3'][query_idx].tolist()

        # now reformatting into final query. this is the entire query provided in the dataset as it is as well
        # as in lowercase letters. furthermore, the query is restructured to be easily fit into the api requests.
        # from list to string
        queries = queries[0]

        # removing spaces after commas
        queries = queries.replace(', ', ',')

        # replacing commas with |
        queries = queries.replace(',', '|')

        # creating lowercase copy
        # queries_lowercase = queries.lower()  # NOT NECESSARY AS QUERY IS NOT CASE-SENSITIVE

        # combining original and lowercase for the final query
        queries_final = queries

        # inserting into dataframe
        df_team['query'] = queries_final

        # exporting
        df_team.to_excel(writer, sheet_name=sheet, index=False)

    # closing writer
    writer.close()

    pass


def RedditStructure():
    """
    This function takes the OddsPortal structure and
    transforms this into separate spreadsheets containing the teams to be
    provided with additional data from Reddit.

    The output is an excel-file containing sheets consisting data for the different teams.

    Requirements:
        - pandas
        - numpy
    """

    # imports and libraries
    import pandas as pd
    import numpy as np

    # importing structured OddsPortal data
    df = pd.read_excel('output//OddsPortal_structured.xlsx', sheet_name='2021-2022')

    # creating season variable
    df.insert(1, 'season', '2021-2022')

    # setting path to destination excel file
    destination_path = 'output//Reddit_structured.xlsx'

    # setting up writer
    writer = pd.ExcelWriter(destination_path, engine='xlsxwriter')

    # fetching teams of the season
    teams = df['home_team'].unique()

    # looping through teams to split them into separate excel sheets
    for idx, team in enumerate(teams):
        # filtering based on teams (if team is home or away)
        team_idx = np.where((df['home_team'] == team) | (df['away_team'] == team))
        df_team = df.loc[team_idx].reset_index(drop=True)

        # creating and inserting team variable
        df_team.insert(3, 'team', team)

        # exporting
        df_team.to_excel(writer, sheet_name=team, index=False)

    writer.close()

    return


def FootballDataMerge():
    """
    This function merges FootballData from 2021-2022 and 2022-2023 for each
    team. The output is an Excel file containing the data for each team.

    Requirements:
        - pandas
        - datetime
    """

    # imports and libraries
    import pandas as pd
    import datetime

    """
    First merging football-data.co.uk data. Now, these are split into seasons
    2021-2022 and 2022-2023 and these need to be in 2021-2023.
    """

    # importing data
    fd_path = 'FD_PATH'
    fd = pd.read_excel(fd_path, sheet_name=None)
    fd_2021_2022 = fd['2021_2022']
    fd_2022_2023 = fd['2022_2023']

    # keeping only columns considered in the analysis as bookmaker odds are from OddsPortal.com
    columns = ['Season',
               'Date',
               'Time',
               'HomeTeam',
               'AwayTeam',
               'FTHG',
               'FTAG',
               'FTR',
               'HTHG',
               'HTAG',
               'HTR',
               'Referee',
               'HS',
               'AS',
               'HST',
               'AST',
               'HC',
               'AC',
               'HF',
               'AF',
               'HY',
               'AY',
               'HR',
               'AR',
               'AvgH',
               'AvgD',
               'AvgA',
               'MaxH',
               'MaxD',
               'MaxA']

    fd_2021_2022 = fd_2021_2022[columns]
    fd_2022_2023 = fd_2022_2023[columns]

    # Manchester City and Manchester United are not named the same across Football
    # Data and OddsPortal. Therefore, changing these in Football Data to fit the
    # merging.
    # 2021-2022
    fd_2021_2022.loc[fd_2021_2022['HomeTeam'] == 'Man City', 'HomeTeam'] = 'Manchester City'
    fd_2021_2022.loc[fd_2021_2022['AwayTeam'] == 'Man City', 'AwayTeam'] = 'Manchester City'
    fd_2021_2022.loc[fd_2021_2022['HomeTeam'] == 'Man United', 'HomeTeam'] = 'Manchester Utd'
    fd_2021_2022.loc[fd_2021_2022['AwayTeam'] == 'Man United', 'AwayTeam'] = 'Manchester Utd'

    # 2022-2023
    fd_2022_2023.loc[fd_2022_2023['HomeTeam'] == 'Man City', 'HomeTeam'] = 'Manchester City'
    fd_2022_2023.loc[fd_2022_2023['AwayTeam'] == 'Man City', 'AwayTeam'] = 'Manchester City'
    fd_2022_2023.loc[fd_2022_2023['HomeTeam'] == 'Man United', 'HomeTeam'] = 'Manchester Utd'
    fd_2022_2023.loc[fd_2022_2023['AwayTeam'] == 'Man United', 'AwayTeam'] = 'Manchester Utd'

    # editing date variable corresponding to the one in OddsPortal data
    # 2021-2022
    dates = []
    for idx, date in enumerate(fd_2021_2022['Date']):
        time = fd_2021_2022['Time'][idx]
        dates.append(datetime.datetime.combine(date, time))

    fd_2021_2022['Date'] = dates
    fd_2021_2022 = fd_2021_2022.drop(columns=['Time'])

    # 2022-2023
    dates = []
    for idx, date in enumerate(fd_2022_2023['Date']):
        time = fd_2022_2023['Time'][idx]
        dates.append(datetime.datetime.combine(date, time))

    fd_2022_2023['Date'] = dates
    fd_2022_2023 = fd_2022_2023.drop(columns=['Time'])

    # combining the two datasets
    fd_2021_2023 = pd.concat((fd_2021_2022, fd_2022_2023)).reset_index(drop=True)

    # now splitting into different sheets and saving in excel
    # setting up destination path
    destination_path = 'DESTINATION_PATH'

    # setting up writer
    writer = pd.ExcelWriter(destination_path, engine='xlsxwriter')

    # extracting teams
    teams = fd_2021_2023['HomeTeam'].unique()

    # splitting into separate teams
    for team in teams:
        # fetching home and away games and concatenating
        fd_team_home = fd_2021_2023[fd_2021_2023['HomeTeam'] == team].reset_index(drop=True)
        fd_team_away = fd_2021_2023[fd_2021_2023['AwayTeam'] == team].reset_index(drop=True)
        fd_team = pd.concat((fd_team_home, fd_team_away)).reset_index(drop=True)

        # sorting chronologically
        fd_team = fd_team.sort_values(by='Date').reset_index(drop=True)

        # creating team varible
        fd_team.insert(2, 'team', team)

        # exporting
        fd_team.to_excel(writer, sheet_name=team, index=False)

    writer.close()

    pass


def OddsPortalMerge():
    """
    This function merges OddsPortal data for 2021-2022 and 2022-2023 for each
    team. The output is an Excel file containing the data for each team.

    Requirements:
        - pandas
    """

    # imports and libraries
    import pandas as pd

    # importing data
    op_2021_2022 = pd.read_excel('OP_2021_2022_PATH', sheet_name=None)
    op_2022_2023 = pd.read_excel('OP_2022_2023_PATH', sheet_name=None)

    # fetching teams for each season
    teams_2021_2022 = list(op_2021_2022.keys())
    teams_2022_2023 = list(op_2022_2023.keys())

    # combining into all teams across seasons
    teams_2021_2023 = teams_2021_2022 + list(set(teams_2022_2023) - set(teams_2021_2022))

    # setting up destination path
    destination_path = 'DESTINATION_PATH'

    # setting up writer
    writer = pd.ExcelWriter(destination_path, engine='xlsxwriter')

    # merging seasons
    for team in teams_2021_2023:
        # if the team is in both seasons
        if team in teams_2021_2022 and team in teams_2022_2023:
            # collecting data for both seasons
            team_2021_2022 = op_2021_2022[team]
            team_2022_2023 = op_2022_2023[team]

            # concatenating
            op_team = pd.concat((team_2021_2022, team_2022_2023)).reset_index(drop=True)

            # sorting chronologically
            op_team = op_team.sort_values(by='date').reset_index(drop=True)

        # if the team is only in 2021-2022 season
        if team in teams_2021_2022 and team not in teams_2022_2023:
            # only data for 2021-2022 season
            op_team = op_2021_2022[team]

            # sorting chronologically
            op_team = op_team.sort_values(by='date').reset_index(drop=True)

        # if the team is only in 2022-2023 season
        if team not in teams_2021_2022 and team in teams_2022_2023:
            # only data for 2022-2023 season
            op_team = op_2022_2023[team]

            # sorting chronologically
            op_team = op_team.sort_values(by='date').reset_index(drop=True)

        # exporting
        op_team.to_excel(writer, sheet_name=team, index=False)

    writer.close()

    pass


def RedditMerge():
    """
    This function merges Reddit data for 2021-2022 and 2022-2023 for each
    team. The output is an Excel file containing the data for each team.

    Requirements:
        - pandas
        - os
    """

    # imports and libraries
    import pandas as pd
    import os

    # setting up data paths (folders)
    reddit_2021_2022_path = 'REDDIT_2021_2022_PATH'
    reddit_2022_2023_path = 'REDDIT_2022_2023_PATH'

    # fetching file names (teams) of each season
    teams_2021_2022 = os.listdir(reddit_2021_2022_path)
    teams_2022_2023 = os.listdir(reddit_2022_2023_path)

    # combining into all teams across seasons
    teams_2021_2023 = teams_2021_2022 + list(set(teams_2022_2023) - set(teams_2021_2022))

    # setting up destination path
    destination_path = 'DESTINATION_PATH'

    # setting up writer
    writer = pd.ExcelWriter(destination_path, engine='xlsxwriter')

    # merging seasons
    for team in teams_2021_2023:

        # progress
        print(team)

        # if the team is in both seasons
        if team in teams_2021_2022 and team in teams_2022_2023:
            # setting team name
            team_name = team.split('.')[0]

            # collecting data for both seasons
            team_2021_2022 = pd.read_excel(f'{reddit_2021_2022_path}//{team}')
            team_2022_2023 = pd.read_excel(f'{reddit_2022_2023_path}//{team}')

            # concatenating
            reddit_team = pd.concat((team_2021_2022, team_2022_2023)).reset_index(drop=True)

            # sorting chronologically
            reddit_team = reddit_team.sort_values(by='date').reset_index(drop=True)

        # if the team is only in 2021-2022 season
        if team in teams_2021_2022 and team not in teams_2022_2023:
            # setting team name
            team_name = team.split('.')[0]

            # only data for 2021-2022 season
            reddit_team = pd.read_excel(f'{reddit_2021_2022_path}//{team}')

            # sorting chronologically
            reddit_team = reddit_team.sort_values(by='date').reset_index(drop=True)

        # if the team is only in the 2022-2023 season
        if team not in teams_2021_2022 and team in teams_2022_2023:
            # setting team name
            team_name = team.split('.')[0]

            # only data for 2022-2023 season
            reddit_team = pd.read_excel(f'{reddit_2022_2023_path}//{team}')

            # sorting chronologically
            reddit_team = reddit_team.sort_values(by='date').reset_index(drop=True)

        # exporting
        reddit_team.to_excel(writer, sheet_name=team_name, index=False)

    writer.close()

    pass


def RedditDropDuplicates():
    """
    This function eliminates duplicate comments for each team based on comment
    id. As this is done on comment id, I make sure that it is the exact same
    comment (same author, same body, same time posted) and therefore this
    comment only needs to be in the dataset once.

    Requirements:
        - pandas
    """

    # imports and libraries
    import pandas as pd

    # importing data
    reddit_data_path = 'REDDIT_DATA_PATH'
    reddit_data = pd.read_excel(reddit_data_path, sheet_name=None)

    # fetching teams
    teams = list(reddit_data.keys())

    # setting up destination path
    destination_path = 'DESTINATION_PATH'

    # setting up writer
    writer = pd.ExcelWriter(destination_path, engine='xlsxwriter')

    for team in teams:
        reddit_team = reddit_data[team]

        # dropping duplicate comments
        reddit_team = reddit_team.drop_duplicates(subset='id', keep='first').reset_index(drop=True)

        # exporting
        reddit_team.to_excel(writer, sheet_name=team, index=False)

    writer.close()

    pass


def DataMerge():
    """
    This function merges the collected data. That is data from OddsPortal,
    FootballData, and Reddit. The output is an Excel file containing the
    merged data split into separate teams.

    Requirements:
        - pandas
    """

    # imports and libraries
    import pandas as pd

    # importing data
    footballdata = pd.read_excel('output//FootballData_split.xlsx', sheet_name=None)
    oddsportal = pd.read_excel('output//OddsPortal_split.xlsx', sheet_name=None)
    reddit = pd.read_excel('output//Reddit_split_nodup_sentiments_aggregate.xlsx', sheet_name=None)

    # setting up destination path
    destination_path = 'output//AllData_split.xlsx'

    # setting up writer
    writer = pd.ExcelWriter(destination_path, engine='xlsxwriter')

    # fetching teams
    teams = list(oddsportal.keys())

    for team in teams:
        print(team)

        # selecting team
        footballdata_team = footballdata[team]
        oddsportal_team = oddsportal[team]
        reddit_team = reddit[team]

        # checking dimensions before
        print(f'Dataframe dimensions: {oddsportal_team.shape}')

        # merging onto oddsportal data
        # footballdata
        all_team = oddsportal_team.merge(right=footballdata_team,
                                         how='left',
                                         left_on=['date', 'team'],
                                         right_on=['Date', 'team'])

        # reddit
        all_team = all_team.merge(right=reddit_team,
                                  how='left',
                                  left_on=['game_id', 'team'],
                                  right_on=['game_id', 'team'])

        # selecting period
        all_team = all_team[~(all_team['date'] > '2022-12-31')]

        # checking dimensions after
        print(f'Dataframe dimensions: {all_team.shape}')

        # exporting
        all_team.to_excel(writer, sheet_name=team, index=False)

    writer.close()

    pass


def AggregateAllData():
    """
    This function aggregates the data and gathers this
    in one sheet ready for analysis.

    Requirements:
        - pandas
    """

    # imports and libraries
    import pandas as pd

    # importing data
    all_data_split = pd.read_excel('output//AllData_split.xlsx', sheet_name=None)

    # fetching teams
    teams = list(all_data_split.keys())

    """
    The function defined below should only be called if one wishes data
    to be edited to the 5 day information set format. In this, the deviation
    from mean sentiment across teams are also calculated.

    Remember to correct all other functions that use this data, so output
    files do not replace files in the other format
    """

    # creating dataframe to contain all data
    all_data = pd.DataFrame()

    # appending individual team data
    for team in teams:
        all_data_split_team = all_data_split[team]
        all_data = all_data.append(all_data_split_team)

    # exporting
    all_data.to_excel('output//AllData.xlsx', sheet_name='AllData', index=False)

    pass


def Rankings_2021_2022():
    """
    This function calculates team rankings of the 2021-2022 season

    Requirements:
        - pandas
        - numpy
        - datetime
    """

    # imports and libraries
    import pandas as pd
    import numpy as np
    import datetime

    # importing data
    data = pd.read_excel('output//AllData.xlsx')

    # sorting after time of match
    data = data.sort_values(by=['date'], ignore_index=True)

    """
    Executing for 2021-2022 season
    """

    data = data[data['season'] == '2021-2022'].reset_index(drop=True)

    # only getting one entry per game
    data = data.drop_duplicates(subset=['game_id'], ignore_index=True)

    # getting teams
    teams = data['HomeTeam'].unique()

    # creating dataframe containg date in range of matches and the teams. this is
    # used for containing points for each team at each date
    # creating list of dates
    start_date = data['date'][0] - datetime.timedelta(days=3)
    end_date = data['date'][len(data['date']) - 1]
    dates = pd.date_range(start=start_date,
                          end=end_date,
                          freq='min')

    # only keeping if minutes are are 00, 15, 30, or 45
    relevant_dates = [date for date in dates if date.minute in [00, 15, 30, 45]]

    # generating dictionary for containing team points
    points = {}
    for team in teams:
        points[team] = [0] * len(relevant_dates)

    # updating points
    for game_idx, game in enumerate(data['game_id']):

        # defining outcome of match
        outcome = str(data['outcome'][game_idx])

        # if home team wins
        if outcome == '1':
            # fetching winner team
            winner = data['HomeTeam'][game_idx]

            # fetching date index
            game_date = data['date'][game_idx]

            # getting index in dates list
            date_index = relevant_dates.index(game_date)

            # inserting points
            points[winner][date_index] = 3

        # if draw
        if outcome == 'X':
            # fetching teams
            home_team = data['HomeTeam'][game_idx]
            away_team = data['AwayTeam'][game_idx]

            # fetching date index
            game_date = data['date'][game_idx]

            # getting index in dates list
            date_index = relevant_dates.index(game_date)

            # inserting points
            points[home_team][date_index] = 1
            points[away_team][date_index] = 1

        # if away team wins
        if outcome == '2':
            # fetching winner team
            winner = data['AwayTeam'][game_idx]

            # fetching date index
            game_date = data['date'][game_idx]

            # getting index in dates list
            date_index = relevant_dates.index(game_date)

            # inserting points
            points[winner][date_index] = 3

    # now updating to sum of points
    points_sum = {}
    for team in teams:
        points_sum[team] = [0] * len(relevant_dates)

    for team in teams:
        for idx in range(len(relevant_dates)):
            # if not first game of team
            if idx != 0:
                points_sum[team][idx] = points_sum[team][idx - 1] + points[team][idx]

    # sorting teams to fit each other
    teams = sorted(teams)
    keys = list(points.keys())
    keys.sort()
    points = {i: points[i] for i in keys}
    points_sum = {i: points_sum[i] for i in keys}

    # creating rankings
    rankings = {}
    for team in teams:
        rankings[team] = [np.nan] * len(relevant_dates)

    # creating dataframe to containing rankings
    df_rankings = pd.DataFrame(columns=teams)

    for date_index, date in enumerate(relevant_dates):
        ranking_date = date

        # getting points of team on date
        points_date = [item[date_index] for item in points_sum.values()]

        # this is now a row containg points across all teams
        # inserting this in dataframe
        df_rankings.loc[date_index] = points_date

    # converting to actual rankings
    df_rankings = df_rankings.astype(int)
    df_rankings = df_rankings.rank(axis=1, ascending=False, method='first')

    # inserting rankings for each game in data
    hometeam_ranking = []
    awayteam_ranking = []

    for date_idx, date in enumerate(data['date']):
        # getting index in rankings dataframe for this date
        game_date = date
        date_index = relevant_dates.index(game_date)

        # getting home and away team
        hometeam = data['HomeTeam'][date_idx]
        awayteam = data['AwayTeam'][date_idx]

        # getting home and away team ranking
        htrank = df_rankings[hometeam][date_index]
        atrank = df_rankings[awayteam][date_index]

        # appending to list containing rankings
        hometeam_ranking.append(int(htrank))
        awayteam_ranking.append(int(atrank))

    return hometeam_ranking, awayteam_ranking


def Rankings_2022_2023():
    """
    This function calculates team rankings of the 2022-2023 season

    Requirements:
        - pandas
        - numpy
        - datetime
    """
    # imports and libraries
    import pandas as pd
    import numpy as np
    import datetime

    # importing data
    data = pd.read_excel('output//AllData.xlsx')

    # sorting after time of match
    data = data.sort_values(by=['date'], ignore_index=True)

    """
    Executing for 2022-2023 season
    """

    data = data[data['season'] == '2022-2023'].reset_index(drop=True)

    # only getting one entry per game
    data = data.drop_duplicates(subset=['game_id'], ignore_index=True)

    # getting teams
    teams = data['HomeTeam'].unique()

    # creating dataframe containg date in range of matches and the teams. this is
    # used for containing points for each team at each date
    # creating list of dates
    start_date = data['date'][0] - datetime.timedelta(days=3)
    end_date = data['date'][len(data['date']) - 1]
    dates = pd.date_range(start=start_date,
                          end=end_date,
                          freq='min')

    # only keeping if minutes are are 00, 15, 30, or 45
    relevant_dates = [date for date in dates if date.minute in [00, 15, 30, 45]]

    # generating dictionary for containing team points
    points = {}
    for team in teams:
        points[team] = [0] * len(relevant_dates)

    # updating points
    for game_idx, game in enumerate(data['game_id']):

        # defining outcome of match
        outcome = str(data['outcome'][game_idx])

        # if home team wins
        if outcome == '1':
            # fetching winner team
            winner = data['HomeTeam'][game_idx]

            # fetching date index
            game_date = data['date'][game_idx]

            # getting index in dates list
            date_index = relevant_dates.index(game_date)

            # inserting points
            points[winner][date_index] = 3

        # if draw
        if outcome == 'X':
            # fetching teams
            home_team = data['HomeTeam'][game_idx]
            away_team = data['AwayTeam'][game_idx]

            # fetching date index
            game_date = data['date'][game_idx]

            # getting index in dates list
            date_index = relevant_dates.index(game_date)

            # inserting points
            points[home_team][date_index] = 1
            points[away_team][date_index] = 1

        # if away team wins
        if outcome == '2':
            # fetching winner team
            winner = data['AwayTeam'][game_idx]

            # fetching date index
            game_date = data['date'][game_idx]

            # getting index in dates list
            date_index = relevant_dates.index(game_date)

            # inserting points
            points[winner][date_index] = 3

    # now updating to sum of points
    points_sum = {}
    for team in teams:
        points_sum[team] = [0] * len(relevant_dates)

    for team in teams:
        for idx in range(len(relevant_dates)):
            # if not first game of team
            if idx != 0:
                points_sum[team][idx] = points_sum[team][idx - 1] + points[team][idx]

    # sorting teams to fit each other
    teams = sorted(teams)
    keys = list(points.keys())
    keys.sort()
    points = {i: points[i] for i in keys}
    points_sum = {i: points_sum[i] for i in keys}

    # creating rankings
    rankings = {}
    for team in teams:
        rankings[team] = [np.nan] * len(relevant_dates)

    # creating dataframe to containing rankings
    df_rankings = pd.DataFrame(columns=teams)

    for date_index, date in enumerate(relevant_dates):
        ranking_date = date

        # getting points of team on date
        points_date = [item[date_index] for item in points_sum.values()]

        # this is now a row containg points across all teams
        # inserting this in dataframe
        df_rankings.loc[date_index] = points_date

    # converting to actual rankings
    df_rankings = df_rankings.astype(int)
    df_rankings = df_rankings.rank(axis=1, ascending=False, method='first')

    # inserting rankings for each game in data
    hometeam_ranking = []
    awayteam_ranking = []

    for date_idx, date in enumerate(data['date']):
        # getting index in rankings dataframe for this date
        game_date = date
        date_index = relevant_dates.index(game_date)

        # getting home and away team
        hometeam = data['HomeTeam'][date_idx]
        awayteam = data['AwayTeam'][date_idx]

        # getting home and away team ranking
        htrank = df_rankings[hometeam][date_index]
        atrank = df_rankings[awayteam][date_index]

        # appending to list containing rankings
        hometeam_ranking.append(int(htrank))
        awayteam_ranking.append(int(atrank))

    return hometeam_ranking, awayteam_ranking


def AggregateRankings():
    """
    This function aggregates rankings of each team in a form to
    be used in the analysis.

    Requirements:
        - pandas
    """

    # imports and libraries
    import pandas as pd

    # importing data
    data = pd.read_excel('output//AllData.xlsx')

    # only getting one entry per game
    data = data.drop_duplicates(subset=['game_id'], ignore_index=True)

    # sorting data by time of match
    data = data.sort_values(by=['date'], ignore_index=True)

    # getting rankings from above functions
    htrank_2021_2022, atrank_2021_2022 = Rankings_2021_2022()
    htrank_2022_2023, atrank_2022_2023 = Rankings_2022_2023()

    # stacking list of rankings
    htrank_2021_2023 = htrank_2021_2022 + htrank_2022_2023
    atrank_2021_2023 = atrank_2021_2022 + atrank_2022_2023

    # inserting in data
    data['HTRank'] = htrank_2021_2023
    data['ATRank'] = atrank_2021_2023

    # exporting
    data.to_excel('output//AllData_rankings.xlsx', sheet_name='Data')

    pass


def NormalizeOdds():
    """
    This function calculates inverse odds, normalized odds and booksums.

    Requirements:
        - pandas
    """


    # imports and libraries
    import pandas as pd

    # importing data
    data = pd.read_excel('output//AllData_rankings.xlsx')

    # defining lists for normalized odds and inverse odds
    OddsPortal_1_inv_norm = []
    OddsPortal_X_inv_norm = []
    OddsPortal_2_inv_norm = []
    OddsPortal_1_inv = []
    OddsPortal_X_inv = []
    OddsPortal_2_inv = []
    OddsPortal_booksums = []

    for game_idx, game in enumerate(data['game_id']):
        # fetching odds for each outcome
        home_odds = data['OddsPortal_1'][game_idx]
        draw_odds = data['OddsPortal_X'][game_idx]
        away_odds = data['OddsPortal_2'][game_idx]

        # calculating inverse odds
        home_odds_inv = 1 / home_odds
        draw_odds_inv = 1 / draw_odds
        away_odds_inv = 1 / away_odds

        # calculating booksum
        booksum = home_odds_inv + draw_odds_inv + away_odds_inv

        # normalizing odds
        home_odds_inv_norm = home_odds_inv / booksum
        draw_odds_inv_norm = draw_odds_inv / booksum
        away_odds_inv_norm = away_odds_inv / booksum

        # inserting in lists
        OddsPortal_1_inv_norm.append(home_odds_inv_norm)
        OddsPortal_X_inv_norm.append(draw_odds_inv_norm)
        OddsPortal_2_inv_norm.append(away_odds_inv_norm)
        OddsPortal_1_inv.append(home_odds_inv)
        OddsPortal_X_inv.append(draw_odds_inv)
        OddsPortal_2_inv.append(away_odds_inv)
        OddsPortal_booksums.append(booksum)

    # inserting back into data
    data['OddsPortal_1_inv_norm'] = OddsPortal_1_inv_norm
    data['OddsPortal_X_inv_norm'] = OddsPortal_X_inv_norm
    data['OddsPortal_2_inv_norm'] = OddsPortal_2_inv_norm
    data['OddsPortal_1_inv'] = OddsPortal_1_inv
    data['OddsPortal_X_inv'] = OddsPortal_X_inv
    data['OddsPortal_2_inv'] = OddsPortal_2_inv
    data['OddsPortal_booksum'] = OddsPortal_booksums

    # exporting
    data.to_excel('output//AllData_rankings_normalized.xlsx', sheet_name='Data')

    pass


def CalculateVariables():
    """
    This function uses data from football-data.co.uk to calculate the team
    variables needed for the analysis.This is done for both 2021-2022
    and 2022-2023 separately

    Requirements:
        - pandas
    """

    # imports and libraries
    import pandas as pd

    # importing data
    all_data_split = pd.read_excel('output//AllData_split.xlsx', sheet_name=None)

    # setting up destination path
    destination_path = 'output//AllData_split_correctvars.xlsx'

    # setting up writer
    writer = pd.ExcelWriter(destination_path, engine='xlsxwriter')

    # fetching teams
    teams = list(all_data_split.keys())

    for team in teams:

        """ 2021-2022 """
        # defining team data
        team_data = all_data_split[team]

        # for 2021-2022 season
        team_data = team_data[team_data['season'] == '2021-2022'].reset_index(drop=True)

        # creating lists for new variables
        goals = []
        shots_target = []
        corners = []
        yellows = []
        reds = []
        goals_against = []  # new
        shots_target_against = []  # new

        for team_index, team in enumerate(team_data['team']):

            # if the team is home team
            if team_data['home_team'][team_index] == team:
                goals.append(team_data['FTHG'][team_index])
                shots_target.append(team_data['HST'][team_index])
                corners.append(team_data['HC'][team_index])
                yellows.append(team_data['HY'][team_index])
                reds.append(team_data['HR'][team_index])
                goals_against.append(team_data['FTAG'][team_index])  # new
                shots_target_against.append(team_data['AST'][team_index])  # new

            # if the team is away team
            if team_data['away_team'][team_index] == team:
                goals.append(team_data['FTAG'][team_index])
                shots_target.append(team_data['AST'][team_index])
                corners.append(team_data['AC'][team_index])
                yellows.append(team_data['AY'][team_index])
                reds.append(team_data['AR'][team_index])
                goals_against.append(team_data['FTHG'][team_index])  # new
                shots_target_against.append(team_data['HST'][team_index])  # new

        # inserting into team dataframe
        team_data['Goals'] = goals
        team_data['ShotsTarget'] = shots_target
        team_data['Corners'] = corners
        team_data['Yellow'] = yellows
        team_data['Red'] = reds
        team_data['GoalsAgainst'] = goals_against  # new
        team_data['ShotsTargetAgainst'] = shots_target_against  # new

        # creating new variables for rolling data (information set of 5 games)

        # defining information set size
        iss = 5

        team_data['Goals_5'] = team_data['Goals'].rolling(window=iss, closed='left').sum()
        team_data['ShotsTarget_5'] = team_data['ShotsTarget'].rolling(window=iss, closed='left').sum()
        team_data['Corners_5'] = team_data['Corners'].rolling(window=iss, closed='left').sum()
        team_data['Yellow_5'] = team_data['Yellow'].rolling(window=iss, closed='left').sum()
        team_data['Red_5'] = team_data['Red'].rolling(window=iss, closed='left').sum()
        team_data['GoalsAgainst_5'] = team_data['GoalsAgainst'].rolling(window=iss, closed='left').sum()  # new
        team_data['ShotsTargetAgainst_5'] = team_data['ShotsTargetAgainst'].rolling(window=iss,
                                                                                    closed='left').sum()  # new

        # generating copy to contain 2021-2022 data before moving on to 2022-2023
        # team_data_dest = team_data.copy(deep=True).values()
        team_data_dest = team_data.to_dict()

        """ 2022-2023 """
        # defining team data
        team_data = all_data_split[team]

        # for 2021-2022 season
        team_data = team_data[team_data['season'] == '2022-2023'].reset_index(drop=True)

        # creating lists for new variables
        goals = []
        shots_target = []
        corners = []
        yellows = []
        reds = []
        goals_against = []  # new
        shots_target_against = []  # new

        for team_index, team in enumerate(team_data['team']):

            # if the team is home team
            if team_data['home_team'][team_index] == team:
                goals.append(team_data['FTHG'][team_index])
                shots_target.append(team_data['HST'][team_index])
                corners.append(team_data['HC'][team_index])
                yellows.append(team_data['HY'][team_index])
                reds.append(team_data['HR'][team_index])
                goals_against.append(team_data['FTAG'][team_index])  # new
                shots_target_against.append(team_data['AST'][team_index])  # new

            # if the team is away team
            if team_data['away_team'][team_index] == team:
                goals.append(team_data['FTAG'][team_index])
                shots_target.append(team_data['AST'][team_index])
                corners.append(team_data['AC'][team_index])
                yellows.append(team_data['AY'][team_index])
                reds.append(team_data['AR'][team_index])
                goals_against.append(team_data['FTHG'][team_index])  # new
                shots_target_against.append(team_data['HST'][team_index])  # new

        # inserting into team dataframe
        team_data['Goals'] = goals
        team_data['ShotsTarget'] = shots_target
        team_data['Corners'] = corners
        team_data['Yellow'] = yellows
        team_data['Red'] = reds
        team_data['GoalsAgainst'] = goals_against  # new
        team_data['ShotsTargetAgainst'] = shots_target_against  # new

        # creating new variables for rolling data (information set of 5 games)

        # defining information set size
        iss = 5

        team_data['Goals_5'] = team_data['Goals'].rolling(window=iss, closed='left').sum()
        team_data['ShotsTarget_5'] = team_data['ShotsTarget'].rolling(window=iss, closed='left').sum()
        team_data['Corners_5'] = team_data['Corners'].rolling(window=iss, closed='left').sum()
        team_data['Yellow_5'] = team_data['Yellow'].rolling(window=iss, closed='left').sum()
        team_data['Red_5'] = team_data['Red'].rolling(window=iss, closed='left').sum()
        team_data['GoalsAgainst_5'] = team_data['GoalsAgainst'].rolling(window=iss, closed='left').sum()  # new
        team_data['ShotsTargetAgainst_5'] = team_data['ShotsTargetAgainst'].rolling(window=iss,
                                                                                    closed='left').sum()  # new

        # concatenating the two seasons
        team_data_2021_2022 = pd.DataFrame(team_data_dest)

        team_data_all = pd.concat((team_data_2021_2022, team_data), ignore_index=True)

        # creating lists for sentiment deviation
        vader_neg_dev = []
        vader_neu_dev = []
        vader_pos_dev = []
        vader_compound_dev = []

        # calculating mean sentiments
        vader_neg_mean = team_data_all['vader_neg'].mean()
        vader_neu_mean = team_data_all['vader_neu'].mean()
        vader_pos_mean = team_data_all['vader_pos'].mean()
        vader_compound_mean = team_data_all['vader_compound'].mean()

        for game_idx, game in enumerate(team_data_all['game_id']):
            # negative sentiment
            vader_neg = team_data_all['vader_neg'][game_idx]
            vader_neg_dev.append(vader_neg - vader_neg_mean)

            # neutral sentiment
            vader_neu = team_data_all['vader_neu'][game_idx]
            vader_neu_dev.append(vader_neu - vader_neu_mean)

            # positive sentiment
            vader_pos = team_data_all['vader_pos'][game_idx]
            vader_pos_dev.append(vader_pos - vader_pos_mean)

            # compound sentiment
            vader_compound = team_data_all['vader_compound'][game_idx]
            vader_compound_dev.append(vader_compound - vader_compound_mean)

        # inserting into dataframe
        team_data_all['vader_neg_dev'] = vader_neg_dev
        team_data_all['vader_neu_dev'] = vader_neu_dev
        team_data_all['vader_pos_dev'] = vader_pos_dev
        team_data_all['vader_compound_dev'] = vader_compound_dev

        # exporting
        team_data_all.to_excel(writer, sheet_name=team, index=False)

    writer.close()

    pass


def CalculatedDataMerge():
    """
    This function takes the variables calculated on and puts them into the
    sheet containing all data

    Requirements:
        - pandas
        - numpy
    """

    # imports and libraries
    import pandas as pd
    import numpy as np

    # importing dataset containing calculated variables
    df_calculations = pd.read_excel('output//AllData_split_correctvars.xlsx', sheet_name=None)

    # importing destination dataset
    df_dest = pd.read_excel('output//AllData_rankings_normalized.xlsx')

    # creating columns in destination dataframe for new data
    new_columns = ['HGoals',
                   'AGoals',
                   'HShotsTarget',
                   'AShotsTarget',
                   'HCorners',
                   'ACorners',
                   'HYellow',
                   'AYellow',
                   'HRed',
                   'ARed',
                   'HGoalsAgainst',  # new
                   'AGoalsAgainst',  # new
                   'HShotsTargetAgainst',  # new
                   'AShotsTargetAgainst',  # new
                   'HGoals_5',
                   'AGoals_5',
                   'HShotsTarget_5',
                   'AShotsTarget_5',
                   'HCorners_5',
                   'ACorners_5',
                   'HYellow_5',
                   'AYellow_5',
                   'HRed_5',
                   'ARed_5',
                   'HGoalsAgainst_5',  # new
                   'AGoalsAgainst_5',  # new
                   'HShotsTargetAgainst_5',  # new
                   'AShotsTargetAgainst_5',  # new
                   'HNegSentimentDev',
                   'HNeuSentimentDev',
                   'HPosSentimentDev',
                   'HCompoundSentimentDev',
                   'ANegSentimentDev',
                   'ANeuSentimentDev',
                   'APosSentimentDev',
                   'ACompoundSentimentDev']

    for column in new_columns:
        df_dest[column] = np.nan

    # fetching teams
    teams = list(df_calculations.keys())

    for game_idx, game in enumerate(df_dest['game_id']):
        # teams playing
        hometeam = df_dest['HomeTeam'][game_idx]
        awayteam = df_dest['AwayTeam'][game_idx]

        # time of match
        match_time = df_dest['date'][game_idx]

        # filling in columns
        # goals
        df_dest['HGoals'][game_idx] = df_calculations[hometeam]['Goals'][
            df_calculations[hometeam].index[df_calculations[hometeam]['date'] == match_time]]
        df_dest['AGoals'][game_idx] = df_calculations[awayteam]['Goals'][
            df_calculations[awayteam].index[df_calculations[awayteam]['date'] == match_time]]

        # shots on target
        df_dest['HShotsTarget'][game_idx] = df_calculations[hometeam]['ShotsTarget'][
            df_calculations[hometeam].index[df_calculations[hometeam]['date'] == match_time]]
        df_dest['AShotsTarget'][game_idx] = df_calculations[awayteam]['ShotsTarget'][
            df_calculations[awayteam].index[df_calculations[awayteam]['date'] == match_time]]

        # corners
        df_dest['HCorners'][game_idx] = df_calculations[hometeam]['Corners'][
            df_calculations[hometeam].index[df_calculations[hometeam]['date'] == match_time]]
        df_dest['ACorners'][game_idx] = df_calculations[awayteam]['Corners'][
            df_calculations[awayteam].index[df_calculations[awayteam]['date'] == match_time]]

        # yellow cards
        df_dest['HYellow'][game_idx] = df_calculations[hometeam]['Yellow'][
            df_calculations[hometeam].index[df_calculations[hometeam]['date'] == match_time]]
        df_dest['AYellow'][game_idx] = df_calculations[awayteam]['Yellow'][
            df_calculations[awayteam].index[df_calculations[awayteam]['date'] == match_time]]

        # red cards
        df_dest['HRed'][game_idx] = df_calculations[hometeam]['Red'][
            df_calculations[hometeam].index[df_calculations[hometeam]['date'] == match_time]]
        df_dest['ARed'][game_idx] = df_calculations[awayteam]['Red'][
            df_calculations[awayteam].index[df_calculations[awayteam]['date'] == match_time]]

        # goals against
        df_dest['HGoalsAgainst'][game_idx] = df_calculations[hometeam]['GoalsAgainst'][
            df_calculations[hometeam].index[df_calculations[hometeam]['date'] == match_time]]
        df_dest['AGoalsAgainst'][game_idx] = df_calculations[awayteam]['GoalsAgainst'][
            df_calculations[awayteam].index[df_calculations[awayteam]['date'] == match_time]]

        # shots on target against
        df_dest['HShotsTargetAgainst'][game_idx] = df_calculations[hometeam]['ShotsTargetAgainst'][
            df_calculations[hometeam].index[df_calculations[hometeam]['date'] == match_time]]
        df_dest['AShotsTargetAgainst'][game_idx] = df_calculations[awayteam]['ShotsTargetAgainst'][
            df_calculations[awayteam].index[df_calculations[awayteam]['date'] == match_time]]

        # goals last 5
        df_dest['HGoals_5'][game_idx] = df_calculations[hometeam]['Goals_5'][
            df_calculations[hometeam].index[df_calculations[hometeam]['date'] == match_time]]
        df_dest['AGoals_5'][game_idx] = df_calculations[awayteam]['Goals_5'][
            df_calculations[awayteam].index[df_calculations[awayteam]['date'] == match_time]]

        # shots on target last 5
        df_dest['HShotsTarget_5'][game_idx] = df_calculations[hometeam]['ShotsTarget_5'][
            df_calculations[hometeam].index[df_calculations[hometeam]['date'] == match_time]]
        df_dest['AShotsTarget_5'][game_idx] = df_calculations[awayteam]['ShotsTarget_5'][
            df_calculations[awayteam].index[df_calculations[awayteam]['date'] == match_time]]

        # corners last 5
        df_dest['HCorners_5'][game_idx] = df_calculations[hometeam]['Corners_5'][
            df_calculations[hometeam].index[df_calculations[hometeam]['date'] == match_time]]
        df_dest['ACorners_5'][game_idx] = df_calculations[awayteam]['Corners_5'][
            df_calculations[awayteam].index[df_calculations[awayteam]['date'] == match_time]]

        # yellow cards last 5
        df_dest['HYellow_5'][game_idx] = df_calculations[hometeam]['Yellow_5'][
            df_calculations[hometeam].index[df_calculations[hometeam]['date'] == match_time]]
        df_dest['AYellow_5'][game_idx] = df_calculations[awayteam]['Yellow_5'][
            df_calculations[awayteam].index[df_calculations[awayteam]['date'] == match_time]]

        # red cards last 5
        df_dest['HRed_5'][game_idx] = df_calculations[hometeam]['Red_5'][
            df_calculations[hometeam].index[df_calculations[hometeam]['date'] == match_time]]
        df_dest['ARed_5'][game_idx] = df_calculations[awayteam]['Red_5'][
            df_calculations[awayteam].index[df_calculations[awayteam]['date'] == match_time]]

        # goals against last 5
        df_dest['HGoalsAgainst_5'][game_idx] = df_calculations[hometeam]['GoalsAgainst_5'][
            df_calculations[hometeam].index[df_calculations[hometeam]['date'] == match_time]]
        df_dest['AGoalsAgainst_5'][game_idx] = df_calculations[awayteam]['GoalsAgainst_5'][
            df_calculations[awayteam].index[df_calculations[awayteam]['date'] == match_time]]

        # shots on target against last 5
        df_dest['HShotsTargetAgainst_5'][game_idx] = df_calculations[hometeam]['ShotsTargetAgainst_5'][
            df_calculations[hometeam].index[df_calculations[hometeam]['date'] == match_time]]
        df_dest['AShotsTargetAgainst_5'][game_idx] = df_calculations[awayteam]['ShotsTargetAgainst_5'][
            df_calculations[awayteam].index[df_calculations[awayteam]['date'] == match_time]]

        # negative sentiment deviation
        df_dest['HNegSentimentDev'][game_idx] = df_calculations[hometeam]['vader_neg_dev'][
            df_calculations[hometeam].index[df_calculations[hometeam]['date'] == match_time]]
        df_dest['ANegSentimentDev'][game_idx] = df_calculations[awayteam]['vader_neg_dev'][
            df_calculations[awayteam].index[df_calculations[awayteam]['date'] == match_time]]

        # neutral sentiment deviation
        df_dest['HNeuSentimentDev'][game_idx] = df_calculations[hometeam]['vader_neu_dev'][
            df_calculations[hometeam].index[df_calculations[hometeam]['date'] == match_time]]
        df_dest['ANeuSentimentDev'][game_idx] = df_calculations[awayteam]['vader_neu_dev'][
            df_calculations[awayteam].index[df_calculations[awayteam]['date'] == match_time]]

        # positive sentiment deviation
        df_dest['HPosSentimentDev'][game_idx] = df_calculations[hometeam]['vader_pos_dev'][
            df_calculations[hometeam].index[df_calculations[hometeam]['date'] == match_time]]
        df_dest['APosSentimentDev'][game_idx] = df_calculations[awayteam]['vader_pos_dev'][
            df_calculations[awayteam].index[df_calculations[awayteam]['date'] == match_time]]

        # compound sentiment deviation
        df_dest['HCompoundSentimentDev'][game_idx] = df_calculations[hometeam]['vader_compound_dev'][
            df_calculations[hometeam].index[df_calculations[hometeam]['date'] == match_time]]
        df_dest['ACompoundSentimentDev'][game_idx] = df_calculations[awayteam]['vader_compound_dev'][
            df_calculations[awayteam].index[df_calculations[awayteam]['date'] == match_time]]

    # exporting
    df_dest.to_excel('output//STATA_QUERY.xlsx', sheet_name='QUERY', index=False)

    pass



