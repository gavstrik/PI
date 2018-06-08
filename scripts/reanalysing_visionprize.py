import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
style.use('ggplot')
plt.rcParams["font.family"] = "Times New Roman"

"""
This python script calculates metaknowledge types
and estimation errors for data received from
Vision Prize, 2012-14, http://www.visionprize.com/.
"""

# make a dictionary of data files in the format
# {filename: [(question number, answer number)]}
dict_data_files = {
    '../data/visionprize_2007.csv': [(6, 5)],
    # '../data/visionprize_2013.csv': [(3, 5)],
}

R1Q6_text = '"What is the likelihood that global average sea level \
             will rise more during this century than the highest level \
             given in the 2007 assessment of the IPCC (0.59m)?"'

RQ4Q3_text = '"What is the likelihood that global average sea level \
              will rise {it:more} during this century than the {it:highest} \
              level given in the draft 2013 assessment of the IPCC (.91m)?"'

# make list of metaknowledge types
types = ['tc', 'fc', 'td', 'fd', 'fa']


# function for making the propper dataframe
def make_dataframe(q):
    # create dataframes
    df = pd.DataFrame(data_file)
    df1 = pd.DataFrame()

    df1['id'] = df['id']
    kwargs = {'Q' + str(q[0]): df['Q' + str(q[0])].values}
    df1 = df1.assign(**kwargs)
    for answer in range(1, q[1] + 1):
        kwargs = {str(answer): df['Q' + str(q[0]) +
                                  'distribution_guess' + str(answer)].values}
        df1 = df1.assign(**kwargs)

    # drop all rows with nan's
    df1 = df1.dropna()

    # print(df1.head())
    return df1


# function for calculating the Kullback-leiber Divergence
def KL(P, Q):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 0.0001
    _P = P + epsilon
    _Q = Q + epsilon
    return np.sum(_P * np.log(_P / _Q))


# function for counting metaknowledge types
def metaknowledge_type(x, y, mpa):
    tc = fc = td = fd = fa = 0
    psychological_state = []
    # make a matrix of answers vs predicted most popular answer:
    u = [[0 for i in range(k)] for i in range(k)]
    y_max = []
    for person, answer in enumerate(y):
        personal_guess = int(x[person])
        predicted_most_popular_choice = max(
            range(len(answer)), key=answer.__getitem__)
        y_max.append(predicted_most_popular_choice)
        u[personal_guess][predicted_most_popular_choice] += 1
        if personal_guess == mpa and predicted_most_popular_choice == mpa:
            tc += 1
            psychological_state.append('tc')
        elif personal_guess != mpa and predicted_most_popular_choice == personal_guess:
            fc += 1
            psychological_state.append('fc')
        elif personal_guess == mpa and predicted_most_popular_choice != personal_guess:
            fd += 1
            psychological_state.append('fd')
        elif personal_guess != mpa and predicted_most_popular_choice != personal_guess:
            if predicted_most_popular_choice == mpa:
                td += 1
                psychological_state.append('td')
            else:
                fa += 1
                psychological_state.append('fa')
    return tc, fc, td, fd, fa, psychological_state, u, y_max


def histo(a, b, c):
    sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})
    categories = ['very unlikely\n(<10%)', 'unlikely\n(<33%)',
                  'about as \n likely as not\n(33-66%)', 'likely\n(>66%)',
                  'very likely\n(>90%)']

    with sns.axes_style("white"):
        # plot details
        bar_width = 0.25
        a_bar_positions = np.arange(len(a))
        b_bar_positions = a_bar_positions + bar_width
        c_bar_positions = b_bar_positions + bar_width

        # fonts
        csfont = {'fontname':'Times New Roman'}
        plt.rcParams["figure.figsize"] = (10, 7.5)
        plt.rcParams['legend.fontsize'] = 14
        plt.rcParams['savefig.dpi'] = 600

        # make bar plots in color
        # ax = plt.bar(a_bar_positions, a, bar_width,
        #              label='actual answers')
        # bx = plt.bar(b_bar_positions, b, bar_width,
        #              label='predicted majority')
        # cx = plt.bar(c_bar_positions, c, bar_width,
        #              label='estimated support')

        # make bar plots in greyscale
        ax = plt.bar(a_bar_positions, a, bar_width, color="0.2",
                     label='actual answers')
        bx = plt.bar(b_bar_positions, b, bar_width, color="0.5",
                     label='predicted majority')
        cx = plt.bar(c_bar_positions, c, bar_width, color="0.8",
                     label='estimated support')

        plt.xticks(a_bar_positions + bar_width, categories,
                   fontname="Times New Roman", fontsize=14)
        plt.yticks(fontname="Times New Roman", fontsize=14)
        plt.ylabel('Percentages', fontname="Times New Roman", fontsize=14)
        plt.legend(loc='best', prop={"family": "Times New Roman"})
        plt.title('"What is the likelihood that global average sea level will rise more during this century\n than the highest level given in the 2007 assessment of the IPCC (0.59m)?" (N=133)',
                  fontname="Times New Roman", fontsize=14)
        # plt.title('"What is the likelihood that global average sea level will rise more during this century\n than the highest level given in the draft 2013 assessment of the IPCC (.91m)?" (N=97)',
        #           fontname="Times New Roman", fontsize=15)
        sns.despine()
        plt.tight_layout()
        # plt.savefig('C:/Users/Robin/Documents/Papers/PI/final/figure_4.png')
        # plt.savefig('C:/Users/Robin/Documents/Papers/PI/final/figure_4_greyscale.png')
        # plt.savefig('C:/Users/Robin/Documents/Papers/PI/final/figure_5.png')
        # plt.savefig('C:/Users/Robin/Documents/Papers/PI/final/figure_5_greyscale.png')
        plt.show()


# for each data file:
for key, qanda in dict_data_files.items():

    # load data file from directory
    data_file = pd.read_csv(key)

    # initialize a dataframe for each question and type
    df_scores = pd.DataFrame(columns=types)

    # go through each question
    for question in range(1, len(qanda) + 1):
        print('\n file', key, 'Question', qanda[question - 1][0],
              'number of answers', qanda[question - 1][1])

        # make dataframe with only the question in question
        df = make_dataframe(qanda[question - 1])

        # extract answers x
        x = df['Q' + str(qanda[question - 1][0])].values
        x[:] = [a - 1 for a in x]  # remember to substract 1 from all answers
        x = x.astype(np.int64)

        # calculate x_bar
        _, counts = np.unique(x, return_counts=True)
        x_bar = np.array([answers/sum(counts) for answers in counts])
        print('\nx_bar =', x_bar)

        # extract predictions y as a N x k numpy array
        k = qanda[question - 1][1]
        y = np.array([df[str(a)].values for a in range(1, k + 1)])
        y = y/100

        # find the mean of the predictions
        y = np.transpose(y)
        y_bar = sum(y)/len(y)
        print('y_bar =', y_bar)

        # find the most popular opinions
        mpa = x_bar.tolist().index(max(x_bar))
        print('\nmost popular opinion:', mpa)

        # calculating the KLD for the average y_bar instead of the
        # individual y's.
        kl_xy_bar = KL(x_bar, y_bar)
        print('\nkl_score, x_bar vs. y_bar :\n',
              np.around(np.array(kl_xy_bar), 3))

        # calculate the whole Kullback-Leiber Divergence for each respondent
        kl_scores = [KL(x_bar, prediction) for prediction in y]
        print('\nprediction scores:\n', np.around(np.array(kl_scores), 2))

        # find the metaknowledge types:
        tc, fc, td, fd, fa, psychological_state, u, y_max = metaknowledge_type(
            x, y, mpa)
        tot = tc + fc + td + fd + fa
        print('\n# true consent:', tc, '\n# false consent:', fc, '\n# true dissent:', td, '\n# false dissent:', fd, '\n# false attr.:', fa, '\ntot=', tot)
        print('Pluralistic ignorance score fd/td:', fd/td)
        print('simple pluralistic ignorance score (fd+fc+fa)/(tc+td):', (fd+fc+fa)/(tc+td))

        # print response matrix
        print('\nresponse matrix:\n', np.matrix(u))

        add_row = []
        for j in range(k):
            add_row.append(sum((u[i][j] for i in range(len(u)))))
        add_row = [i for i in add_row]
        print('\ntotal for each predicted opinion:\n', np.array(add_row)/tot)

        # do some plotting:
        histo(np.array(x_bar), np.array(add_row)/tot, np.array(y_bar))

        upper_sum = sum(sum((u[i][i+1:] for i in range(len(u)) ), []))
        lower_sum = sum(sum((u[i][:i] for i in range(len(u)) ), []))
        print('\nupper_sum = ', upper_sum)
        print('lower_sum = ', lower_sum)
        print('conservative bias score = ', lower_sum/upper_sum)
        print('Taylor\'s index = ', (fa+fd+fc)/tot)

        # find the average estimated support to each opinion by m-types:
        print('\nmean estimated support to each opinion by m-types:')
        kl_mtypes = []
        for i in types:
            nums = len([psychological_state for j in psychological_state if j == i])
            y_dummy = [estimate for pos, estimate in enumerate(y)
                       if psychological_state[pos] == i]
            y_T_dummy = np.transpose(y_dummy)
            est_mtypes = np.array([sum(y_T_dummy[i]) for i in range(k)])/nums
            kl_mtypes.append(KL(x_bar, est_mtypes))
            print(i, nums, est_mtypes)
            # print('kl-score of collective estimates by', i, KL(x_bar, est_mtypes))

        # find the mean prediction error for all:
        print('\nmean prediction error (all) = \n', sum(kl_scores)/len(kl_scores))

        # find the mean prediction error for each opinion group:
        print('\nmean prediction error (opinion groups) = ')
        for i in range(k):
            mpe_og = [score for pos, score in enumerate(kl_scores)
                      if x[pos] == i]
            mpe_og = sum(mpe_og)/len(mpe_og)
            print(i, mpe_og)

        # find the mean prediction error for each group predicting the same:
        print('\nmean prediction error (prediction groups) = ')
        for i in range(k):
            mpe_pg = [score for pos, score in enumerate(kl_scores)
                           if y_max[pos] == i]
            mpe_pg = sum(mpe_pg)/len(mpe_pg)
            print(i, mpe_pg)

        # find the mean prediction error for each mtype:
        print('\nmean prediction error (mtypes) = ')
        for i in types:
            mpe_mt = [score for pos, score in enumerate(kl_scores)
                           if psychological_state[pos] == i]
            mpe_mt = sum(mpe_mt)/len(mpe_mt)
            print(i, mpe_mt)

        # find the collective prediction error for all:
        y_T = np.transpose(y)
        y_bar_new = np.array([sum(y_T[i]) for i in range(k)])/tot
        cpe_all = KL(x_bar, y_bar_new)
        print('\ncollective prediction error (all) =\n', cpe_all)

        # find the collective prediction error by opinion group:
        print('\ncollective prediction error (opinion groups) =')
        for i in range(k):
            nums = len([x for j in x if j == i])
            y_dummy = [estimate for pos, estimate in enumerate(y) if x[pos] == i]
            y_T_dummy = np.transpose(y_dummy)
            est_op_group = np.array([sum(y_T_dummy[i]) for i in range(k)])/nums
            cpe_op = KL(x_bar, est_op_group)
            print(i, cpe_op)

        # find the average estimated support to each opinion by m-type:
        print('\ncollective prediction error (mtypes) =')
        for i in types:
            nums = len([psychological_state for j in psychological_state if j == i])
            y_dummy = [estimate for pos, estimate in enumerate(y)
                       if psychological_state[pos] == i]
            y_T_dummy = np.transpose(y_dummy)
            est_mtype = np.array([sum(y_T_dummy[i]) for i in range(k)])/nums
            cpe_mt = KL(x_bar, est_mtype)
            print(i, cpe_mt)
