from numpy.random import normal
from math import sqrt, e
from numpy import array, median, amax

number_of_group_expertise = 10
number_of_experts = 6
evaluation_methods = {"mean":0,
                      "median":0,
                      "iterative":0,
                      "method of lowest squares": 0,
                      "method of lowest absolutes": 0}
evaluation_methods_errors = {"mean":0,
                      "median":0,
                      "iterative":0,
                      "method of lowest squares": 0,
                      "method of lowest absolutes": 0}
evaluation_methods_dispersion = {"mean":0,
                      "median":0,
                      "iterative":0,
                      "method of lowest squares": 0,
                      "method of lowest absolutes": 0}
E = 0.001


def gen_data(dist_params, with_anom=False, with_ref=False, c=5, anom_c=9, ref_params=(3, 1)):
    """ Generate experts evaluations """

    samples, data, u = [], [], []

    if with_ref:
        u = normal(*ref_params, size=number_of_group_expertise)

    for p in dist_params:
        for _ in range(p[2]):
            samples.append(normal(p[0], p[1], size=number_of_group_expertise))

    if with_anom:
        samples.append(number_of_group_expertise * [anom_c])

    for i in range(number_of_group_expertise):
        for sample in samples:

            if with_anom and sample is samples[-1]:
                number = sample[i]
            else:
                if with_ref:
                    number = int(round(u[i] + sample[i]))
                else:
                    number = int(round(c + sample[i]))

            if number > 10:
                data.append(10)
            elif number < 0:
                data.append(0)
            else:
                data.append(number)

    sigma_xi(data, u, with_ref)
    return data, u


def count_mean(data_in_matrix):
    mean_for_every_object = []
    for row in data_in_matrix:
        sum = 0
        for grade in row:
            sum += grade
        mean_for_every_object.append(sum / number_of_experts)
    return mean_for_every_object


def method_lowest_squares(data_in_matrix, mean_grades_for_method):
    mlq_grades = []
    row_counter = 0
    for row in data_in_matrix:
        sum_for_every_row = 0
        for grade in row:
            sum_for_every_row += (grade - mean_grades_for_method[row_counter]) ** 2
        row_counter += 1
        if sum_for_every_row > 10:
            sum_for_every_row = 10
        mlq_grades.append(sum_for_every_row)
    return mlq_grades


def method_lowest_abs(data_in_matrix):
    mla_grades = []
    for row in data_in_matrix:
        sum = 0
        for grade in row:
            sum += abs(grade - median(row))
        mla_grades.append(sum)
    return mla_grades


def distance(data_in_matrix, center):
    return [sqrt(sum([(data_in_matrix[i][j] - center[i]) ** 2 for i in range(data_in_matrix.shape[0])])) for j in range(data_in_matrix.shape[1])]


def n_distance(data_in_matrix, distances):
    return [i / ((amax(data_in_matrix) - 1) * sqrt(data_in_matrix.shape[0])) for i in distances]


def comp_level(distances, b=0.967, b0=15):
    return [1 / ((1 - b) * e ** (b0 * i) + b) for i in distances]


def iterative_eval(row_in_matrix, c=0.05):
    x = sum(row_in_matrix) / len(row_in_matrix)
    w = [1 / abs(i - x) if abs(i - x) else 1 / c for i in row_in_matrix]
    count = 1

    while True:
        x_prev = x
        x = sum([w[i] * row_in_matrix[i] for i in range(len(w))]) / sum(w)
        w = [1 / abs(i - x) if abs(i - x) else 1 / c for i in row_in_matrix]
        count += 1

        if abs(x - x_prev) <= E:
            break

    print('x = {0:.4f}, number of iterations: {1}, for data: {2}'.format(x, count, row_in_matrix))
    return x


def sigma_xi(data_in_matrix, u, ref, c=5):
    if ref:
        s = 0
        data_in_matrix = to_matrix(data_in_matrix, number_of_group_expertise)

        for j in u:
            for i in data_in_matrix:
                for k in i:
                    s += (k - j) * (k - j)

        print('sigma_xi = {0:.4f}'.format(sqrt(s / len(data_in_matrix))))

    else:
        print('sigma_xi = {0:.4f}'.format(sqrt(sum([(data_in_matrix[i] - c) * (data_in_matrix[i] - c)
                                                    for i in range(len(data_in_matrix))]) / len(data_in_matrix))))


def errors_for_method(dict, u):
    for key in dict:
        error_for_method = []
        for j in range(len(dict[key])):
            error = dict[key][j] - u[j]
            error_for_method.append(error)
        evaluation_methods_errors[key] = error_for_method
    return


def cov(errors_for_each_method):
    dict_of_tuples = get_tuples_of_methods(errors_for_each_method)
    for key in dict_of_tuples:
        cov = 0
        for i in range(len(evaluation_methods_errors[key[0]])):
            cov += evaluation_methods_errors[key[0]][i] * evaluation_methods_errors[key[1]][i]
        dict_of_tuples[key] = 1 / (len(list(evaluation_methods_errors.keys()))) * cov
    return dict_of_tuples


def get_tuples_of_methods(dict_of_methods):
    list_of_methods = list(dict_of_methods.keys())
    dict_of_tuples = {}
    for i in range(len(list_of_methods) - 1):
        for j in range(i + 1, len(list_of_methods)):
            tuple_of_methods = (list_of_methods[i], list_of_methods[j])
            dict_of_tuples[tuple_of_methods] = 0
    return dict_of_tuples


def covariation_matrix(dict_of_dispersions, dict_of_cov):
    matrix = []
    print("Length of dispersion: " + str(len(list(dict_of_dispersions.keys()))))
    print("Length of cov: " + str(len(list(dict_of_cov.keys()))))
    return


def mid_square_error(dict_of_errors):
    dict_of_dispersion = {}
    for key in dict_of_errors:
        dispersion = 0
        for i in range(len(dict_of_errors[key])):
            dispersion += dict_of_errors[key][i] ** 2
        dict_of_dispersion[key] = (1 / len(list(dict_of_errors.keys()))) * dispersion
    return dict_of_dispersion


def count_median(data):
    med_grades = []
    for i in data:
        med_grade = median(i)
        med_grades.append(med_grade)
    return med_grades


def to_matrix(a, n):
    """ Split list into n parts and return data with parts like rows """
    print("List: " + str(a))
    k, m = divmod(len(a), n)
    return array([a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)])


def run():
    t = [(0, 2, number_of_experts)]

    # with given parameters
    # tours(t,ref=False, s_dist=True)
    # with given parameters and reference values
    data, u = gen_data(t, with_anom=False, with_ref=True)
    data = to_matrix(data, number_of_group_expertise)
    mean_grades = count_mean(data)
    evaluation_methods["mean"] = mean_grades
    median_grades = count_median(data)
    evaluation_methods["median"] = median_grades
    mls = method_lowest_squares(data, mean_grades)
    evaluation_methods["method of lowest squares"] = mls
    mla = method_lowest_abs(data)
    evaluation_methods["method of lowest absolutes"] = mla
    iterative_method = [iterative_eval(row) for row in data]
    evaluation_methods["iterative"] = iterative_method
    # data_transposed = data.transpose()
    dist = distance(data, mean_grades)
    n_dist = n_distance(data, dist)
    competence_evaluation = comp_level(n_dist)
    # transposed_matrix = data.transpose()
    print(str(data))
    print(amax(data))
    print("Mean for every object of experise: " + str(mean_grades))
    print("Median for every object of experise: " + str(median_grades))
    print("Method of lowest squares for every object of expertise: " + str(mls))
    print("Method of lowest absolute for every object of expertise: " + str(mla))
    print("Iterative method for every object of expertise: " + str(iterative_method))
    print("Competence level:" + str(competence_evaluation))
    errors_for_method(evaluation_methods, u)
    print(str(evaluation_methods_errors))
    # TODO make cov for more 2 methods (use dictionary and loops to make structure)
    dict_cov = cov(evaluation_methods_errors)
    print(str(cov(evaluation_methods_errors)))
    dispersion_dict = mid_square_error(evaluation_methods_errors)
    print(str(dispersion_dict))
    covariation_matrix(dispersion_dict, dict_cov)



if __name__ == '__main__':
   run()
   print("Runninug")



