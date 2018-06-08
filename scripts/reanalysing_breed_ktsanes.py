import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
style.use('ggplot')
plt.rcParams["font.family"] = "Times New Roman"

"""
This python script calculates metaknowledge types
and estimation errors for data from Breed, W. and
T. Ktsanes (1961). "Pluralistic ignorance in the
process of opinion formation.", Public Opinion
Quarterly 25(3): 382-392.
"""


class Constants():
    N = 122  # number of respondents
    k = 5  # number of response categories
    types = ['tc', 'fc', 'td', 'fd', 'fa']  # list of metaknowledge types


def opinions():  # opinion data is imported directly from the paper:
    quit_church = [0] * 23
    protest = [1] * 29
    uncertain = [2] * 11
    misgivings = [3] * 39
    welcome = [4] * 20

    # make opinion array x in order of decreasing "conservatims":
    x = quit_church + protest + uncertain + misgivings + welcome
    return np.array(x)


def predictions():  # prediction data is also imported from paper:
    y = []
    for _ in range(13):
        y.append([1, 0, 0, 0, 0])
    for _ in range(10):
        y.append([0, 1, 0, 0, 0])
    for _ in range(3):
        y.append([1, 0, 0, 0, 0])
    for _ in range(24):
        y.append([0, 1, 0, 0, 0])
    for _ in range(2):
        y.append([0, 0, 1, 0, 0])
    for _ in range(1):
        y.append([1, 0, 0, 0, 0])
    for _ in range(6):
        y.append([0, 1, 0, 0, 0])
    for _ in range(3):
        y.append([0, 0, 1, 0, 0])
    for _ in range(1):
        y.append([0, 0, 0, 1, 0])
    for _ in range(2):
        y.append([1, 0, 0, 0, 0])
    for _ in range(19):
        y.append([0, 1, 0, 0, 0])
    for _ in range(7):
        y.append([0, 0, 1, 0, 0])
    for _ in range(11):
        y.append([0, 0, 0, 1, 0])
    for _ in range(2):
        y.append([1, 0, 0, 0, 0])
    for _ in range(5):
        y.append([0, 1, 0, 0, 0])
    for _ in range(7):
        y.append([0, 0, 1, 0, 0])
    for _ in range(4):
        y.append([0, 0, 0, 1, 0])
    for _ in range(2):
        y.append([0, 0, 0, 0, 1])
    return np.array(y)


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


# plotting stuff
def histo(a, b):
    sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})
    categories = ['Quit the church\n entirely.',
                  'Feel bad about\n the decision and\n protest against it.',
                  'Feel uncertain.',
                  'Accept the deci-\nsion, but with\n some misgivings.',
                  'Accept and wel-\n come the change.']

    with sns.axes_style("white"):
        # plot details
        bar_width = 0.35
        a_bar_positions = np.arange(len(a))
        b_bar_positions = a_bar_positions + bar_width

        # fonts
        csfont = {'fontname':'Times New Roman'}
        plt.rcParams["figure.figsize"] = (10, 7.5)
        plt.rcParams['legend.fontsize'] = 14
        plt.rcParams['savefig.dpi'] = 600

        # make bar plots in color
        ax = plt.bar(a_bar_positions, a, bar_width, label='actual answers')
        bx = plt.bar(b_bar_positions, b, bar_width, label='predicted majority')

        # alternatively: make bar plots in greyscale
        # ax = plt.bar(a_bar_positions, a, bar_width, color="0.2",
        #              label='actual answers')
        # bx = plt.bar(b_bar_positions, b, bar_width, color="0.5",
        #              label='predicted majority')

        # paraphernalia
        plt.xticks(a_bar_positions + bar_width/2, categories,
                   fontname="Times New Roman", fontsize=14)
        plt.yticks(fontname="Times New Roman", fontsize=14)
        plt.ylabel('Percentages', fontname="Times New Roman", fontsize=14)
        plt.legend(loc='best', prop={"family": "Times New Roman"})
        # plt.title('"Suppose, for the moment, that the congregation of the ... Church has\n voted to let Negroes join the church. What would you do? (N=122)"')
        plt.title('Reactions to racial desegregation of the church (N=122)',
                   fontname="Times New Roman", fontsize=15)
        sns.despine()
        plt.tight_layout()
        plt.savefig('C:/Users/Robin/Documents/Papers/PI/final/figure_1.png')
        # plt.savefig('C:/Users/Robin/Documents/Papers/PI/final/figure_1_greyscale.png')
        plt.show()


# find the frequencies of opinions, x_bar:
x = opinions()
c = Counter(x)
x_bar = np.array([value/len(x) for key, value in sorted(c.items())])
print('x_bar =', x_bar)

# find the frequencies of predictions, y_bar:
y = predictions()
y_bar = sum(y)/len(y)
print('y_bar =', y_bar)

# do some plotting:
histo(x_bar, y_bar)

# find the most popular opinion
mpa = x_bar.tolist().index(max(x_bar))
print('most popular opinion is in position', mpa)

# calculating the KLD for the average y_bar instead of the
# individual y's:
kl_xy_bar = KL(x_bar, y_bar)
print('\nkl_score, x_bar vs. y_bar :\n',
      np.around(np.array(kl_xy_bar), 3))

# calculate the Kullback-Leiber Divergence for each respondent
# kl_scores = [KL(x_bar, prediction) for prediction in y]
# print('\nestimation errors:\n', np.around(np.array(kl_scores), 2))

# find the metaknowledge types:
tc, fc, td, fd, fa, psychological_state, u, y_max = metaknowledge_type(x, y, mpa)
tot = tc + fc + td + fd + fa
print('\n# true consent:', tc, 100*tc/tot, '%\n# false consent:', fc, 100*fc/tot,
      '%\n# true dissent:', td, 100*td/tot, '%\n# false dissent:', fd, 100*fd/tot,
      '%\n# false attr.:', fa, 100*fa/tot, '%\nchecksum=', tot)

# print response matrix
print('\nresponse matrix:\n', np.matrix(u))

upper_sum = sum(sum((u[i][i+1:] for i in range(len(u))), []))
lower_sum = sum(sum((u[i][:i] for i in range(len(u))), []))
print('\nupper_sum = ', upper_sum)
print('lower_sum = ', lower_sum)
print('conservative bias score = ', lower_sum/upper_sum)
print('Taylor\'s index = ', (fa+fd+fc)/tot)
