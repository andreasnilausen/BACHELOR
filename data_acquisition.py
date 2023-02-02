
def OddsPortal():
    """
    This function accesses and pulls historical data of the 2021-2022 English Premier League
    football season. This is done through OddsPortal (https://www.oddsportal.com/).

    OddsPortal.com collects and provides average betting odds from several online bookmakers
    for all games in the sample.

    The scraping is done using Selenium. The output is stored as a Microsoft Excel spreadsheet (.xlsx).
    The spreadsheet contains 8 sheets each containing the corresponding data from OddsPortal. The output
    data is not yet structured. This is done in 'OddsPortal_structure.py'.

    Requirements:
        - pandas
        - selenium
    """
    # imports
    import pandas as pd
    from selenium import webdriver
    from selenium.webdriver.firefox.options import Options

    # creating dataframe for all pages containing results
    pages = []

    for page in range(1, 9):  # 8 pages in total, looping through these
        # setting url
        url = f'https://www.oddsportal.com/soccer/england/premier-league-2021-2022/results/#/page/{page}/'

        # initializing webdriver
        options = Options()
        browser = webdriver.Firefox(options=options)
        browser.maximize_window()
        browser.get(url)

        # accessing table and converting to dataframe
        table = browser.find_element_by_xpath('//*[@id="tournamentTable"]')
        table_html = table.get_attribute('outerHTML')
        df_page = pd.read_html(table_html)[0]

        # closing current browser
        browser.close()

        # correcting columns as they are now multiindex
        df_page.columns = df_page.columns.get_level_values(1)

        # saving page in list
        pages.append(df_page)

    # exporting each page as separate Excel sheets in same file
    destination_path = 'output//oddsportal_raw.xlsx'
    with pd.ExcelWriter(destination_path) as writer:
        for index, page in enumerate(pages):
            page.to_excel(writer, sheet_name=f'page{index + 1}', index=False)

    pass


def Reddit_v6():
    """
    This function pulls data from Reddit using the PMAW-api (https://pypi.org/project/pmaw/). This script
    saves each sheet with teams if errors occur, so it is not needed to run through all teams again.

    Testing first with 2 games for each team

    Instead of shifting the window, I try to pull the entire window at once when not setting a limit. If there is too
    little data from here, it instead shifts the window 24 hours at a time with a limit of 400 comments.

    Requirements:
        - pandas
        - pmaw
        - datetime
        - os
    """

    # imports and libraries
    import pandas as pd
    from pmaw import PushshiftAPI
    import datetime
    import time

    # importing data
    df = pd.read_excel('output//Reddit_structured_searchquery.xlsx', sheet_name=None)

    # storing sheet names (teams) in list
    teams = list(df.keys())

    """
    WARNING: THE BELOW (UNTIL NEXT COMMENT) SHOULD ONLY BE EXECUTED TO RUN THE PROGRAM ON A SUBSET OF DATA.
    RUN 1:
        - ALL
    RUN 2: 
        - INCLUDED UP TO BUT NOT INCLUDING WOLVES (17/20).
        - THEREFORE STILL MISSING INDICES 16-19
    RUN 3:
        - ERROR IN SOUTHAMPTON QUERY
        - INCLUDING UP TO BUT NOT INCLUDING SOUTHAMPTON
        - THEREFORE STILL MISSING INDICES 17-19
    
    """
    print(teams)

    # keeping only subset that is missing
    teams = teams[17:]

    print(teams)

    # keeping only subset of teams for testing
    # teams = teams[:2]

    """
    WARNING CODE ENDS HERE
    """

    # setting path to destination excel file
    destination_path = 'output//Reddit_comments//Reddit_comments.xlsx'

    # defining subreddits to search
    subreddits = 'soccer,football,PremierLeague'

    # looping through each team
    for team_idx, team in enumerate(teams):

        # initializing api object (new for each team)
        api = PushshiftAPI(num_workers=15)

        # monitoring execution time
        team_start_time = time.time()

        # loading dataframe containing given team
        df_team = df[team]

        # keeping only subset of data for testing
        # df_team = df_team[:2].reset_index(drop=True)

        # creating copy of dataframe for storing data with comments
        df_dest = df_team

        # creating lists for to contain subsets containing comments
        frames = []

        # looping through each match of team
        for match_idx, game in enumerate(df_team['game_id']):

            match_start_time = time.time()

            # checking progress
            print(f'{team} ({team_idx + 1}/{len(teams)}), Match {match_idx + 1}/{len(df_team)}')

            # accessing time of match
            time_of_match = df_team['date'][match_idx]

            """
            we want data for 96 hours leading up to the match. therefore i create a windows inside which pulls should
            be made
            """
            # defining timespan variable
            timespan = 96  # hours

            # setting up before and after for the api request. this creates a window of 96
            # hours to pull the data in
            before_epoch = int(time_of_match.timestamp())
            after_epoch = int((time_of_match - datetime.timedelta(hours=timespan)).timestamp())

            # requesting comments
            comments = api.search_comments(q=df_team['query'][match_idx],  # search query
                                           subreddit=subreddits,  # subreddits to search
                                           mem_safe=True,  # ensuring no memory overload
                                           # limit=400,  # limit of comments to return (for memory), disabled
                                           sort='desc',
                                           sort_type='created_utc',
                                           after=after_epoch,
                                           before=before_epoch)

            # filtering comments to only contain relevant keys
            comments_list = [comment for comment in comments]

            fields = ['created_utc', 'id', 'is_submitter', 'permalink', 'score',
                      'body', 'subreddit', 'subreddit_id', 'retrieved_utc',
                      'updated_utc', 'utc_datetime_str']

            comments_filtered = []
            for comment in comments_list:
                filtered = dict((k, comment[k]) for k in fields)
                comments_filtered.append(filtered)

            # creating temporary dataframe to be merged with the original
            df_comments = pd.DataFrame(comments_filtered)

            # creating game_id variable for the merge
            df_comments['game_id'] = df_team['game_id'][match_idx]

            # checking if no data in dataframe
            if df_comments.shape[0] < 10:  # if less than 10 comments accessed (less than 10 rows)

                print('No results from 96 hour window. Trying with split window.')

                """
                again i want data for 96 hours leading up to the match. to try pulling data
                in this window i do this in 4 different days instead of pulling all in one
                request.
                """

                # creating timespan variables
                timespan = 24  # hours
                timespan_sum = 0

                while timespan_sum < 96:  # 96 hours, 4 days leading up to the match
                    # setting up before and after for api request. this creates a
                    # window of 24 hours i want to pull the within.
                    before_epoch = int((time_of_match - datetime.timedelta(hours=timespan_sum)).timestamp())
                    after_epoch = int((time_of_match - datetime.timedelta(hours=timespan_sum + timespan)).timestamp())

                    # requesting comments
                    comments = api.search_comments(q=df_team['query'][match_idx],  # search query
                                                   subreddit=subreddits,  # subreddits to search
                                                   mem_safe=True,  # ensuring no memory overload
                                                   limit=400,  # limit of comments to return (for memory)
                                                   sort='desc',
                                                   sort_type='created_utc',
                                                   after=after_epoch,
                                                   before=before_epoch)

                    # filtering comments to only contain relevant keys
                    comments_list = [comment for comment in comments]

                    fields = ['created_utc', 'id', 'is_submitter', 'permalink', 'score',
                              'body', 'subreddit', 'subreddit_id', 'retrieved_utc',
                              'updated_utc', 'utc_datetime_str']

                    comments_filtered = []
                    for comment in comments_list:
                        filtered = dict((k, comment[k]) for k in fields)
                        comments_filtered.append(filtered)

                    # creating temporary dataframe to be merged with the original
                    df_comments = pd.DataFrame(comments_filtered)

                    # creating game_id variable for the merge
                    df_comments['game_id'] = df_team['game_id'][match_idx]

                    # inserting into frame of subsets
                    frames.append(df_comments)

                    # exporting as .csv as backup
                    df_comments.to_csv(f'output//Reddit_comments//games//{team}_{game}.csv', index=False)

                    # updating timespan
                    timespan_sum += timespan  # moving on to the previous day

            # inserting into frame of subsets
            frames.append(df_comments)

            # exporting as .csv as backup
            df_comments.to_csv(f'output//Reddit_comments//games//{team}_{game}.csv', index=False)

            # concatenating subset dataframes
            df_frames = pd.concat(frames)

            match_end_time = time.time()

            print(f'Match execution time: {match_end_time - match_start_time}s.')

        # merging with original dataframe
        df_dest = pd.merge(df_dest, df_frames, how='left', on='game_id')

        # exporting
        df_dest.to_excel(f'output//Reddit_comments//{team}.xlsx', sheet_name=team, index=False)

        team_end_time = time.time()

        print(f'Team execution time: {team_end_time - team_start_time}s.')

        pass


