

def NaiveModelPredictions():
    """
    This function calculates predictions of the two naive models.

    Requirements:
        - pandas
        - numpy
    """


    # imports and libraries
    import pandas as pd
    import numpy as np

    # importing data
    data_path = 'DATA_PATH'
    data = pd.read_excel(data_path)

    """
    Calculating results for model 8: The naive model always predicting a home win
    """
    # arbitrary prediction probabilites
    data['p1model8'] = 1
    data['p2model8'] = 0
    data['p3model8'] = 0
    data['prediction_model8'] = 1
    correct_model8 = [np.nan] * len(data)

    for idx, outcome in enumerate(data['probit_outcome']):

        if outcome == 1:
            correct_model8[idx] = 1

        else:
            correct_model8[idx] = 0

    data['correct_model8'] = correct_model8

    # calculating model 8 accurracy
    data_2022_2023 = data[data['season'] == '2022-2023'].reset_index(drop=True)
    model8_acc = data_2022_2023['correct_model8'].mean()

    """
    Calculating results for model 9: The bookmaker model, always betting on the 
    team with the lowest bookmaker odds
    """
    prediction_model9 = [np.nan] * len(data)
    correct_model9 = [np.nan] * len(data)

    for idx, game in enumerate(data['game_id']):

        odds_1 = data['OddsPortal_1'][idx]
        odds_X = data['OddsPortal_X'][idx]
        odds_2 = data['OddsPortal_2'][idx]

        # inserting odds in list
        odds = [odds_1, odds_X, odds_2]

        if min(odds) == odds_1:
            prediction_model9[idx] = 1

        if min(odds) == odds_X:
            prediction_model9[idx] = 2

        if min(odds) == odds_2:
            prediction_model9[idx] = 3

    for idx, outcome in enumerate(data['probit_outcome']):

        if outcome == prediction_model9[idx]:
            correct_model9[idx] = 1

        else:
            correct_model9[idx] = 0

    # arbitrary prediction probabilities
    p1model9 = [np.nan] * len(data)
    p2model9 = [np.nan] * len(data)
    p3model9 = [np.nan] * len(data)

    for idx, prediction in enumerate(prediction_model9):
        if prediction == 1:
            p1model9[idx] = 1
            p2model9[idx] = 0
            p3model9[idx] = 0

        if prediction == 2:
            p1model9[idx] = 0
            p2model9[idx] = 1
            p3model9[idx] = 0

        if prediction == 3:
            p1model9[idx] = 0
            p2model9[idx] = 0
            p3model9[idx] = 1

    data['p1model9'] = p1model9
    data['p2model9'] = p2model9
    data['p3model9'] = p3model9
    data['prediction_model9'] = prediction_model9
    data['correct_model9'] = correct_model9

    # calculating model 9 accurracy
    data_2022_2023 = data[data['season'] == '2022-2023'].reset_index(drop=True)
    model9_acc = data_2022_2023['correct_model9'].mean()

    # exporting
    export_path = 'EXPORT_PATH'
    data.to_excel(export_path, sheet_name='Data', index=False)

    pass


def BrierScores():
    """
    This function calculates the Brier score of each model. Again, only based
    on forecasting period 2022-2023 season.

    Requirements:
        - pandas
        - numpy
    """

    # imports and libraries
    import pandas as pd
    import numpy as np

    # importing data
    data_path = 'DATA_PATH'
    data = pd.read_excel(data_path)

    # only calculating brier scores for forecasting period
    data = data[data['season'] == '2022-2023']

    # creating dictionary for brier scores
    brier_scores = {}

    # defining function for calculating brier score
    def brier_calc(targets, probs):

        devs = []

        for idx1, val1 in enumerate(targets):

            for idx2, val2 in enumerate(val1):
                squared_dev = (probs[idx1][idx2] - val2) ** 2

                devs.append(squared_dev)

        return np.mean(devs)

    # defining true probabilites
    y_true = []
    for idx, outcome in enumerate(data['probit_outcome']):

        if outcome == 1:
            y_true.append([1, 0, 0])

        if outcome == 2:
            y_true.append([0, 1, 0])

        if outcome == 3:
            y_true.append([0, 0, 1])

    # calculating brier score for each model
    for model in range(1, 10):

        # defining predicted probabilities
        y_prob = []
        for prob1, prob2, prob3 in zip(data[f'p1model{str(model)}'], data[f'p2model{str(model)}'],
                                       data[f'p3model{str(model)}']):
            y_prob.append([prob1, prob2, prob3])

        # calculating brier score
        brier_score = brier_calc(y_true, y_prob)
        brier_scores.update({f'Model {str(model)}': brier_score})

    pass


def BettingProfits():
    """
    This function calculates betting profits based on the predictions made
    by the models.

    Requirements:
        - pandas
    """

    # imports and libraries
    import pandas as pd

    # importing data
    data_path = 'DATA_PATH'
    data = pd.read_excel(data_path)

    # calculating betting profits for each model
    profits = {}

    for model in range(1, 10):

        # data[f'profits_model{str(model)}'] = np.nan
        betting_revenue = []

        data = data[data['season'] == '2022-2023'].reset_index(drop=True)

        for idx, correct_pred in enumerate(data[f'correct_model{str(model)}']):

            if int(correct_pred) == 1:

                if data['probit_outcome'][idx] == 1:
                    betting_revenue.append(float(data['OddsPortal_1'][idx]) - 1)

                if data['probit_outcome'][idx] == 2:
                    betting_revenue.append(float(data['OddsPortal_X'][idx]) - 1)

                if data['probit_outcome'][idx] == 3:
                    betting_revenue.append(float(data['OddsPortal_2'][idx]) - 1)


            else:

                betting_revenue.append(-1)

        data[f'profits_model{str(model)}'] = betting_revenue

        profits.update({f'Model {str(model)}': sum(betting_revenue)})

    pass


def CrossTabulation():
    """
    This function calculates the cross tabulation tables for each model.

    Requirements:
        - pandas
        - numpy
    """
    # imports and libraries
    import pandas as pd
    import numpy as np

    # importing data
    data_path = 'DATA_PATH'
    data = pd.read_excel(data_path)

    # keeping only forecasting season
    data = data[data['season'] == '2022-2023'].reset_index(drop=True)

    y_true = np.array(data['probit_outcome'].tolist())
    y_pred = np.array(data['prediction_model9'].tolist())

    print(pd.crosstab(y_true, y_pred, dropna=False))

    pass


def ModelCorrelations():
    """
    This function calculates the correlations between the models and
    the bookmaker predictions

    Requirements:
        - pandas
    """
    # imports and libraries
    import pandas as pd

    # importing data
    data_path = 'DATA_PATH'
    data = pd.read_excel(data_path)

    # keeping only forecasting period
    data = data[data['season'] == '2022-2023'].reset_index(drop=True)

    for model in range(1, 10):
        correlation = data[f'prediction_model{str(model)}'].corr(data['prediction_model9'])

        print(f'Correlation M{str(model)}: {correlation:.3f}')

    pass


def OddsPortalMeanDifferenceTest():
    """
    Function for testing difference in means in OddsPortal data. This is
    done as a paired t-test, as the odds are not independent.

    Requirements:
        - pandas
        - scipy
    """

    # imports and libraries
    import pandas as pd
    from scipy.stats import ttest_rel

    # importing data
    data = pd.read_excel('output//AllData.xlsx')

    # dropping duplicates as only one entry per game is needed
    data = data.drop_duplicates('game_id').reset_index(drop=True)

    # defining samples
    home_wins = data['OddsPortal_1']
    draws = data['OddsPortal_X']
    away_wins = data['OddsPortal_2']

    # performing paired two sample t-tests
    print('Home Wins vs. Away Wins')
    print(ttest_rel(home_wins, away_wins))

    print('Home Wins vs. Draws')
    print(ttest_rel(home_wins, draws))

    print('Draws vs. Away Wins')
    print(ttest_rel(draws, away_wins))

    pass


def OddsPortal_boxplot():
    """
    This function creates a boxplot of the average odds provided
    by OddsPortal.com.

    Requirements:
        - pandas
        - matplotlib
    """

    # imports and libraries
    import pandas as pd
    import matplotlib.pyplot as plt

    # importing data
    data = pd.read_excel('output//AllData.xlsx')

    # dropping duplicates as only one entry per game is needed
    data = data.drop_duplicates('game_id').reset_index(drop=True)

    # creating boxplot
    plt.style.use('default')

    plot = data.boxplot(['OddsPortal_1', 'OddsPortal_X', 'OddsPortal_2'],
                        showfliers=False)

    plot.set_xticklabels(['Home Win', 'Draw', 'Away Win'])
    plot.set_ylabel('Odds')

    # exporting boxplot
    fig = plot.get_figure()
    fig.savefig('output//plots//OddsPortal_boxplot.png')

    # getting data used to create boxplot
    results = ['Home Win', 'Draw', 'Away Win']
    observations = [data['OddsPortal_1'].count(), data['OddsPortal_X'].count(), data['OddsPortal_2'].count()]
    means = [data['OddsPortal_1'].mean(), data['OddsPortal_X'].mean(), data['OddsPortal_2'].mean()]
    standard_deviations = [data['OddsPortal_1'].std(), data['OddsPortal_X'].std(), data['OddsPortal_2'].std()]
    minimums = [data['OddsPortal_1'].min(), data['OddsPortal_X'].min(), data['OddsPortal_2'].min()]
    q1s = [data['OddsPortal_1'].quantile(0.25), data['OddsPortal_X'].quantile(0.25),
           data['OddsPortal_2'].quantile(0.25)]
    q2s = [data['OddsPortal_1'].quantile(0.50), data['OddsPortal_X'].quantile(0.50),
           data['OddsPortal_2'].quantile(0.50)]
    q3s = [data['OddsPortal_1'].quantile(0.75), data['OddsPortal_X'].quantile(0.75),
           data['OddsPortal_2'].quantile(0.75)]
    maximums = [data['OddsPortal_1'].max(), data['OddsPortal_X'].max(), data['OddsPortal_2'].max()]

    # inserting in dataframe and exporting
    data = {'Results': results,
            'Observations': observations,
            'Mean': means,
            'Standard Deviation': standard_deviations,
            'Minimum': minimums,
            'Q1': q1s,
            'Median': q2s,
            'Q3': q3s,
            'Maximum': maximums}

    table = pd.DataFrame(data)

    # exporting table
    table.to_excel('output//tables//OddsPortal_descriptives.xlsx', sheet_name='OddsPortal Descriptives', index=False)

    pass


def OddsPortal_histograms():
    """
    This function creates a boxplot of the average odds provided
    by OddsPortal.com.

    Requirements:
        - pandas
        - matplotlib
    """

    # imports and libraries
    import pandas as pd
    import matplotlib.pyplot as plt

    # importing data
    data = pd.read_excel('output//AllData.xlsx')

    # dropping duplicates as only one entry per game is needed
    data = data.drop_duplicates('game_id').reset_index(drop=True)

    # creating box plot
    plt.style.use('default')
    plt.rcParams["figure.figsize"] = [7.50, 4.50]
    plt.rcParams["figure.autolayout"] = True
    fig, axes = plt.subplots(1, 3)

    # defining number of bins
    bins = 40

    # Home Wins
    histogram_1 = data.hist('OddsPortal_1',
                            ax=axes[0],
                            bins=bins,
                            range=[0, 30])

    for ax in histogram_1.flatten():
        ax.set_title('Home Win')
        ax.set_xlabel('Odds')
        ax.set_ylabel('Matches')
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 300)

    # Draws
    histogram_X = data.hist('OddsPortal_X',
                            ax=axes[1],
                            bins=bins,
                            range=[0, 30])

    for ax in histogram_X.flatten():
        ax.set_title('Draw')
        ax.set_xlabel('Odds')
        ax.set_ylabel('Matches')
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 300)

    # Away Wins
    histogram_2 = data.hist('OddsPortal_2',
                            ax=axes[2],
                            bins=bins,
                            range=[0, 30])

    for ax in histogram_2.flatten():
        ax.set_title('Away Win')
        ax.set_xlabel('Odds')
        ax.set_ylabel('Matches')
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 300)

    plt.show()

    pass


def RedditComments_barplot():
    """
    This function plots the number of comments from Reddit.

    Requirements:
        - pandas
        - matplotlib
    """

    # imports and libraries
    import pandas as pd
    import matplotlib.pyplot as plt

    # importing data
    data = pd.read_excel('output//RedditCommentsCount.xlsx')

    # sorting data
    data = data.sort_values(by=['2021-2023'], ascending=False, ignore_index=True)

    # creating bar plot
    plt.style.use('default')
    plt.rcParams["figure.figsize"] = [7.50, 4.50]
    plt.rcParams["figure.autolayout"] = True
    plot = data[['2021-2022', '2022-2023']].plot.bar(stacked=True)
    plot.set_xticklabels(data['team'], rotation=90)
    plot.set_ylabel('Number of Comments')
    plot.legend(loc='best')
    plt.show()

    # exporting
    fig = plot.get_figure()
    fig.savefig('output//plots//RedditComments_barplot.png')

    pass


def BigFiveRevenue_lineplot():
    """
    This function plots the revenue of the Big Five european football leagues.

    Requirements:
        - pandas
        - matplotlib
    """

    # imports and libraries
    import pandas as pd
    import matplotlib.pyplot as plt

    # importing data
    data_path = 'DATA_PATH'
    data = pd.read_excel(data_path)

    # creating plot
    plt.style.use('default')
    plt.rcParams["figure.figsize"] = [8.50, 4.50]
    plt.rcParams["figure.autolayout"] = True
    plot = data.plot.line()
    plt.xticks(data['Season'].index, rotation=90)
    plot.set_ylim(0, 7500)
    plot.set_xticklabels(data['Season'].tolist(), rotation=90)
    plot.set_ylabel('Revenue in EUR million')
    plot.legend(loc='best')
    plt.show()

    fig = plot.get_figure()
    fig.savefig('output//plots//bigfiverevenue_lineplot.png')

    pass


def AverageSentiment_barplot():
    """
    This function creates the bar plot containing average compound sentiment scores.

    Requirements:
        - pandas
        - matplotlib
    """

    # imports and libraries
    import pandas as pd
    import matplotlib.pyplot as plt

    # importing data
    data_path = 'DATA_PATH'
    data = pd.read_excel(data_path, sheet_name='AverageSentiment')

    # sorting data
    data = data.sort_values(by=['vader_compound_mean'], ascending=False)

    # creating bar plot
    plt.style.use('default')
    plt.rcParams["figure.figsize"] = [7.50, 4.50]
    plt.rcParams["figure.autolayout"] = True
    plot = data['vader_compound_mean'].plot.bar(stacked=True)
    plot.set_xticklabels(data['team'], rotation=90)
    plot.set_ylabel('Average Compound Sentiment')
    plt.show()

    # exporting
    fig = plot.get_figure()
    fig.savefig('output//plots//AverageSentiment_barplot.png')

    pass


def TransformedVariablesHistograms():
    """
    This function calculates histograms of the data used in the analysis.

    Requirements:
        - pandas
        - matplotlib
    """

    # imports and libraries
    import pandas as pd
    import matplotlib.pyplot as plt

    # importing data
    data_path = 'DATA_PATH'
    data = pd.read_excel(data_path)

    # defining variables to be plotted
    variables = ['HTGoals_5',
                 'ATGoals_5',
                 'HTShotsTarget_5',
                 'ATShotsTarget_5',
                 'HTCorners_5',
                 'ATCorners_5',
                 'HTYellow_5',
                 'ATYellow_5',
                 'HTRed_5',
                 'ATRed_5',
                 'HTGoalsAgainst_5',
                 'ATGoalsAgainst_5',
                 'HTShotsTargetAgainst_5',
                 'ATShotsTargetAgainst_5',
                 'HTCompoundSentimentDev',
                 'ATCompoundSentimentDev',
                 'HTRank',
                 'ATRank']

    data = data[variables]

    # creating function to round to nearest five
    def roundtonextfive(series):
        series_max = max(series)
        next_five = series_max + (5 - series_max) % 5
        return next_five

    # creating histogram and kde (edit for each variable)
    plt.style.use('default')
    plt.rcParams["figure.figsize"] = [7.50, 4.50]
    plt.rcParams["figure.autolayout"] = True

    variable = 'ATRank'

    variable_data = data[variable]
    plot = variable_data.hist(bins=20)
    # variable_data.plot.kde(secondary_y=True)
    plt.xlim(0, 21)  # roundtonextfive(variable_data))
    label_size = 25

    # exporting
    fig = plot.get_figure()
    fig.savefig(f'output//plots//{variable}.png')

    pass


def RedditCommentsCount():
    """
    This function counts the number of comments for each teams across seasons
    and prints this

    Requirements:
        - pandas
    """

    # imports and libraries
    import pandas as pd

    # importing dataset with no duplicates
    reddit_data_path = 'REDDIT_DATA_PATH'
    reddit_data = pd.read_excel(reddit_data_path, sheet_name=None)

    # fetching teams
    teams = list(reddit_data.keys())

    # creating lists for data counts
    team_name = []
    n_2021_2022 = []
    n_2022_2023 = []
    n_2021_2023 = []

    for team in teams:
        reddit_team = reddit_data[team]

        # counting number of matches in season
        matches = reddit_team.groupby(['season']).count()

        # inserting in lists
        team_name.append(team)

        # teams only in 2021-2022 season
        if team in ['Norwich', 'Burnley', 'Watford']:
            n_2021_2022.append(matches['game_id'][0])
            n_2022_2023.append(0)
            n_2021_2023.append(matches['game_id'][0])

        # teams only in 2022-2023 season
        elif team in ['Fulham', 'Bournemouth', 'Nottingham']:
            n_2021_2022.append(0)
            n_2022_2023.append(matches['game_id'][0])
            n_2021_2023.append(matches['game_id'][0])

        # all other teams
        else:
            n_2021_2022.append(matches['game_id'][0])
            n_2022_2023.append(matches['game_id'][1])
            n_2021_2023.append(matches['game_id'][0] + matches['game_id'][1])

    # inserting in dataframe
    data = {'team': team_name,
            '2021-2022': n_2021_2022,
            '2022-2023': n_2022_2023,
            '2021-2023': n_2021_2023}

    comments_count = pd.DataFrame(data)

    # exporting
    comments_count.to_excel('output//RedditCommentsCount.xlsx', sheet_name='Number of Comments', index=False)

    pass


def SentimentScores():
    """
    This function calculates sentiment on the Reddit comments via the VADER
    sentiment analysis model. The output is an Excel file containing each
    comment and sentiment scores for each team

    Requirements:
        - pandas
        - vaderSentiment
        - numpy

    """

    # imports and libraries
    import pandas as pd
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import numpy as np

    # importing data
    comments_path = 'output//Reddit_split_nodup.xlsx'
    comments_teams = pd.read_excel(comments_path, sheet_name=None)

    # initializing VADER object
    sia = SentimentIntensityAnalyzer()

    """
    CHANGING THE VALUE OF THE WORD 'UNITED' FROM 1.8 TO 0 AS THIS IS ASSUMED
    TO BE A NEUTRAL WORD POINT TO THE TEAM "MANCHESTER UNITED"
    """
    # changing value of word 'united'
    sia.lexicon['united'] = 0

    # setting up destination path
    destination_path = 'output//Reddit_split_nodup_sentiments.xlsx'

    # setting up writer
    writer = pd.ExcelWriter(destination_path, engine='xlsxwriter')

    # fetching teams
    teams = list(comments_teams.keys())

    for team in teams:

        print(team)

        comments_team = comments_teams[team]

        # defining lists for sentiment scores
        neg_scores, neu_scores, pos_scores, compound_scores = [], [], [], []

        # going through each comment
        for comment in comments_team['body']:

            # if comment is missing
            if pd.isna(comment):

                # inserting missing values
                neg_scores.append(np.nan)
                neu_scores.append(np.nan)
                pos_scores.append(np.nan)
                compound_scores.append(np.nan)

            else:

                # calculating sentiments
                sentiment_scores = sia.polarity_scores(comment)

                # appending to lists
                neg_scores.append(sentiment_scores['neg'])
                neu_scores.append(sentiment_scores['neu'])
                pos_scores.append(sentiment_scores['pos'])
                compound_scores.append(sentiment_scores['compound'])

        # inserting back into data
        comments_team['vader_neg'] = neg_scores
        comments_team['vader_neu'] = neu_scores
        comments_team['vader_pos'] = pos_scores
        comments_team['vader_compound'] = compound_scores

        # exporting
        comments_team.to_excel(writer, sheet_name=team, index=False)

    writer.close()

    pass


def AggregateSentiment():
    """
    This function aggregates the sentiment scores for each match and for each
    team. This is defined as the mean of each of the sentiment scores.

    Requirements:
        - pandas
    """

    # imports and libraries
    import pandas as pd

    # importing data
    comments_path = 'output//Reddit_split_nodup_sentiments.xlsx'
    comments_teams = pd.read_excel(comments_path, sheet_name=None)

    # setting up destination path
    destination_path = 'output//Reddit_split_nodup_sentiments_aggregate.xlsx'

    # setting up writer
    writer = pd.ExcelWriter(destination_path, engine='xlsxwriter')

    # fetching teams
    teams = list(comments_teams.keys())

    for team in teams:
        comments_team = comments_teams[team]

        # calculating mean sentiments for each game
        mean_sentiments = comments_team.groupby(['game_id'], as_index=False).mean()

        # defining columns to keep
        columns = ['game_id', 'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']

        # only keeping defined columns
        mean_sentiments = mean_sentiments[columns]

        # inserting team column
        mean_sentiments.insert(1, 'team', team)

        # exporting
        mean_sentiments.to_excel(writer, sheet_name=team, index=False)

    writer.close()

    pass


def SentimentMeans():
    """
    This function calculates the mean sentiments across teams. The output is
    an Excel file containing mean sentiments across the four categories for
    each team

    Requirements:
        - pandas
    """

    # imports and libraries
    import pandas as pd

    # importing data
    data = pd.read_excel('output//AllData_split.xlsx', sheet_name=None)

    # fetching team names
    teams = list(data.keys())

    # creating lists for variables
    team_names, vader_neg, vader_neu, vader_pos, vader_compound = [], [], [], [], []

    for team in teams:
        data_team = data[team]

        # defining relevant columns
        columns = ['team', 'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']

        # calculating mean columns
        data_means = data_team[columns].groupby(by=['team']).mean()

        # inserting data into lists
        team_names.append(team)
        vader_neg.append(data_means['vader_neg'][0])
        vader_neu.append(data_means['vader_neu'][0])
        vader_pos.append(data_means['vader_pos'][0])
        vader_compound.append(data_means['vader_compound'][0])

    # inserting lists into dataframe
    sentiment_data = {'team': team_names,
                      'vader_neg_mean': vader_neg,
                      'vader_neu_mean': vader_neu,
                      'vader_pos_mean': vader_pos,
                      'vader_compound_mean': vader_compound}

    sentiment_dataframe = pd.DataFrame(sentiment_data)

    # exporting
    sentiment_dataframe.to_excel('output//SentimentMeans.xlsx', sheet_name='Mean Sentiments', index=False)

    pass



