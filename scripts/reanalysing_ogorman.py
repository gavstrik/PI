import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
style.use('ggplot')
plt.rcParams["font.family"] = "Times New Roman"

"""
This python script calculates metaknowledge types
and estimation errors for data used in O'gorman,
H. J. (1975). "Pluralistic ignorance and white
estimates of white support for racial segregation."
Public Opinion Quarterly 39(3): 313-330.
The data is retrieved from the ANES data
set for 1968, see
http://www.electionstudies.org/studypages/download/datacenter_all_datasets.php.

Please note that there is a slight discrepancy in the
number of valid respondes used here to number of
responses used by O'Gorman.
"""


class Constants():
    data_file = pd.read_csv('../data/ogorman.csv')
    types = ['tc', 'fc', 'td', 'fd', 'fa']  # list of metaknowledge types
    k = 3


# function for making the propper dataframe
def make_dataframe(q):
    # create dataframes
    df = pd.DataFrame(q)
    print('all entries: ', len(df))

    # only keep white People
    df = df[df['V680003'] == 2]
    print('only white: ', len(df))

    """
    Page 371: "The sample was truncated to exclude blacks. Also excluded were
    those whites who did not express a racial preference and did not
    estimate the proportions of whites in favor of segregation.
    These respondents, coded as "no answer" or "don't know" on both items,
    constituted 12 per cent of the white sample."
    """

    # remove the question about what blacks think and the color column:
    df = df.drop(columns=['V680086', 'V680003'])

    """
    V680088:
    Q. 33. WHAT ABOUT YOU? ARE YOU IN FAVOR OF DESEGREGATION,
    STRICT SEGREGATION, OR SOMETHING IN BETWEEN?
    ...........................................................
    545 1. DESEGREGATION
    699 3. IN BETWEEN
    236 5. SEGREGATION
    16 8. DK
    61 9. NA
    """
    # only keep people who are either for (0), against(1) or inbetween(2)
    # desegregation
    df['V680088'].replace(1, 0, inplace=True)
    df['V680088'].replace(5, 1, inplace=True)
    df['V680088'].replace(3, 2, inplace=True)
    df['V680088'].replace(8, np.nan, inplace=True)
    df['V680088'].replace(9, np.nan, inplace=True)
    df.dropna(inplace=True)
    print('only those whites who have an opinion: ', len(df))

    """
    V680087:
    Q. 32. HOW ABOUT WHITE PEOPLE IN THIS
    AREA? HOW MANY WOULD YOU SAY ARE IN FAVOR OF
    STRICT SEGREGATION OF THE RACES -- ALL OF THEM,
    MOST OF THEM, ABOUT HALF, LESS THAN HALF OF THEM,
    OR NONE OF THEM?
    .........................................................
    142 1. ALL
    500 2. MOST
    334 3. ABOUT HALF
    317 4. LESS THAN HALF
    68 5. NONE
    158 8. DK
    15 9. NA
    23 0. NO WHITES IN THIS AREA.
    """
    # Page 317: "For purposes of this analysis those who described "all" or
    # "most" whites in their area as segregationists are of primary interest,
    # and the percentage who gave these estimates is used to measure the
    # perceived segregationist majority"
    # eg: replace 1 and 2 with 1,
    #     replace 4 and 5 with 0,
    #     replace 3 with 2
    df['V680087'].replace(0, np.nan, inplace=True) # have to be removed first
    df.dropna(inplace=True)
    print('removed those zeros unaccounted for: ', len(df))

    df['V680087'].replace(1, 1, inplace=True)
    df['V680087'].replace(2, 1, inplace=True)
    df['V680087'].replace(4, 0, inplace=True)
    df['V680087'].replace(5, 0, inplace=True)
    df['V680087'].replace(3, 2, inplace=True)
    df['V680087'].replace(8, np.nan, inplace=True)
    df['V680087'].replace(9, np.nan, inplace=True)

    df.dropna(inplace=True)

    print('only people who are either for, against or in-between: ', len(df))

    return df


def KL(P, Q):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 0.0001
    _P = P + epsilon
    _Q = Q + epsilon
    return np.sum(_P * np.log(_P / _Q))


def metaknowledge_type(x, y, mpa):
    tc = fc = td = fd = fa = 0
    psychological_state = []
    # make a matrix of answers vs predicted most popular answer:
    u = [[0 for i in range(Constants.k)] for i in range(Constants.k)]
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


def histo(a, b):
    sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})
    categories = ['I favor\ndesegre-\ngation', 'all or most\nare for de-\nsegregation',
                   'I favor\n segre-\ngation', 'all or most\nare for se-\ngregation',
                   'I am in\n between', 'half for\nhalf against']

    with sns.axes_style("white"):
        # plot details
        bar_width = 0.35
        a_bar_positions = np.arange(len(a))
        b_bar_positions = a_bar_positions + bar_width + 0.05

        # fonts
        csfont = {'fontname':'Times New Roman'}
        plt.rcParams["figure.figsize"] = (10, 7.5)
        plt.rcParams['legend.fontsize'] = 14
        plt.rcParams['savefig.dpi'] = 600

        # make bar plots in color
        ax = plt.bar(a_bar_positions, a, bar_width, label='actual answers')
        bx = plt.bar(b_bar_positions, b, bar_width, label='predicted majority')

        # make bar plots in greyscale
        # ax = plt.bar(a_bar_positions, a, bar_width, color="0.2",
        #              label='actual answers')
        # bx = plt.bar(b_bar_positions, b, bar_width, color="0.5",
        #              label='predicted majority')

        # paraphernalia
        plt.xticks([0,.4,1,1.4,2,2.4], categories,
                   fontname="Times New Roman", fontsize=14)
        plt.yticks(fontname="Times New Roman", fontsize=14)
        plt.ylabel('Percentages', fontname="Times New Roman", fontsize=14)
        plt.legend(loc='best', prop={"family": "Times New Roman"})
        plt.title('Preferences of racial policy (N=1207)',
                  fontname="Times New Roman", fontsize=15)
        sns.despine()
        plt.tight_layout()
        plt.savefig('C:/Users/Robin/Documents/Papers/PI/final/figure_2.png')
        # plt.savefig('C:/Users/Robin/Documents/Papers/PI/final/figure_2_greyscale.png')
        plt.show()

# extract opinions
df = make_dataframe(Constants.data_file)
x = df['V680088'].values

# calculate x_bar
unique, counts = np.unique(x, return_counts=True)
x_bar = np.array([answers/sum(counts) for answers in counts])
print('x_bar =', x_bar)

# extract predictions
yt = df['V680087'].values
y = []
for i in yt:
    if i == 0:
        y.append([1,0,0])
    elif i == 1:
        y.append([0,1,0])
    else:
        y.append([0,0,1])

y = np.array(y)
y = y.astype(np.float32)
y_bar = sum(y)/len(y)
print('y_bar =', y_bar)

# do some plotting:
histo(x_bar, y_bar)

# find the most popular opinion
mpa = x_bar.tolist().index(max(x_bar))
print('\nmost popular opinion is in position', mpa)

kl_xy_bar = KL(x_bar, y_bar)
print('\nkl_score, x_bar vs. y_bar :\n',
      np.around(np.array(kl_xy_bar), 3))

# calculate the whole Kullback-Leiber Divergence for each respondent
# kl_scores = [KL(x_bar, prediction) for prediction in y]
# print('\nestimation errors:\n', np.around(np.array(kl_scores), 2))

# find the metaknowledge types:
tc, fc, td, fd, fa, psychological_state, u, y_max = metaknowledge_type(x, y, mpa)
tot = tc + fc + td + fd + fa
print('\n# true consent:', tc, '\n# false consent:', fc, '\n# true dissent:', td, '\n# false dissent:', fd, '\n# false attr.:', fa, '\ntot=', tot)

# print responsematrix
print('\nresponse matrix:\n', np.matrix(u))

upper_sum = sum(sum((u[i][i+1:] for i in range(len(u))), []))
lower_sum = sum(sum((u[i][:i] for i in range(len(u))), []))
print('\nupper_sum = ', upper_sum)
print('lower_sum = ', lower_sum)
print('conservative bias score = ', lower_sum/upper_sum)
print('Taylor\'s index = ', (fa+fd+fc)/tot)
