import pandas as pd
import numpy as np

from scipy import stats
from tabulate import tabulate

import matplotlib.pyplot as plt


def data_loading():
    """
    Load the data from the excel file and preprocess it.

    Returns:
    - data (pd.DataFrame): Preprocessed data.
    """
    # Load the data from the excel file
    data = pd.read_excel('gasstationdata33.xlsx')

    # Drop the 'Unnamed: 0' column
    data = data.drop(columns=['Unnamed: 0'])

    data = pd.DataFrame(data)

    arrivals_datetime = [arrival.to_pydatetime() for arrival in data["Arrival Time"]]

    data["Interarrival Times"] = data["Arrival Time"].diff().dt.total_seconds().fillna(0)

    # Calculate the hour of the day
    data['Arrival Time (H)'] = data['Arrival Time'].dt.hour

    # replace nan values in Parking preference with None
    data['Parking preference'] = data['Parking Preference'].fillna('None')

    # fix this mask
    data["Shope-time-no-zero"] = data[data["Shop time"] > 0]["Shop time"]

    # Convert the attributes to minutes
    attributes = ['Service time Fuel', 'Shop time', 'Service time payment', 'Interarrival Times', 'Shope-time-no-zero']

    return data


def calculate_statistics(data, attributes):
    """
    Calculate the test statistics such as mean, std , skewness, kurtosis, median, mode, range, min, max, 25th percentile, 50th percentile, 75th percentile of the payment service time.

    Args:
    - data (pd.DataFrame): Preprocessed data.

    Returns:
    - statistics (pd.DataFrame): Dataframe of summary statistics.
    """
    # Calculate the test statistics

    statistics = [{
        'Mean': data[attribute].mean(),
        'Std': data[attribute].std(),
        'Skewness': data[attribute].skew(),
        'Kurtosis': data[attribute].kurtosis(),
        'Median': data[attribute].median(),
        'Mode': data[attribute].mode().values[0],
        'Range': data[attribute].max() - data[attribute].min(),
        'Min': data[attribute].min(),
        'Max': data[attribute].max(),
        '25th Percentile': data[attribute].quantile(0.25),
        '50th Percentile': data[attribute].quantile(0.50),
        '75th Percentile': data[attribute].quantile(0.75)
    } for attribute in attributes if attribute != 'Shop time']
    shop_data = data[data["Shop time"] > 0]
    statistics.append(
        {
            'Mean': shop_data["Shop time"].mean(),
            'Std': shop_data["Shop time"].std(),
            'Skewness': shop_data["Shop time"].skew(),
            'Kurtosis': shop_data["Shop time"].kurtosis(),
            'Median': shop_data["Shop time"].median(),
            'Mode': shop_data["Shop time"].mode().values[0],
            'Range': shop_data["Shop time"].max() - shop_data["Shop time"].min(),
            'Min': shop_data["Shop time"].min(),
            'Max': shop_data["Shop time"].max(),
            '25th Percentile': shop_data["Shop time"].quantile(0.25),
            '50th Percentile': shop_data["Shop time"].quantile(0.50),
            '75th Percentile': shop_data["Shop time"].quantile(0.75)
        }
    )
    # Convert the statistics list to a DataFrame for easier manipulation
    statistics = pd.DataFrame(statistics, index=attributes).T
    # round to 4 sf
    statistics = statistics.round(4)

    return tabulate(statistics, headers='keys', tablefmt='latex')


def calculate_shop_probabilities(data):
    """
    Calculate the probabilities of having a shop time and the conditional probabilities of having a shop time or not.

    Parameters:
    - data (pd.DataFrame): DataFrame containing shop times.

    Returns:
    - probabilities (dict): Dictionary containing calculated probabilities.
    """
    shop_time = data["Shop time"]
    total_len = len(shop_time)
    has_shop_time = shop_time[shop_time > 0]

    probabilities = {
        "probability_of_shop_time": len(has_shop_time) / total_len,
        "probability_of_no_shop_time": 1 - len(has_shop_time) / total_len,
        "count_total": total_len,
        "count_shop_time": len(has_shop_time),
    }

    return probabilities


def test_binomial_distribution(probabilities, alternative="two-sided", p_value=0.05):
    """
    Test for binomial distribution of shop time.

    Parameters:
    - probabilities (dict): Dictionary containing calculated probabilities.
    - total_len (int): Total number of shop times.

    Returns:
    - test_results (dict): Dictionary containing test results.
    """
    test_results = {
        "test_for_shop_time_binom": stats.binomtest(
            probabilities["count_shop_time"],
            probabilities["count_total"],
            p=probabilities["probability_of_shop_time"],
            alternative=alternative
        ),
    }

    formatted_results = [{"Test": key, "Result": value, "Significance": value.pvalue >= p_value} for key, value in
                         test_results.items()]

    # Use tabulate to format the output
    print(tabulate(formatted_results, headers="keys", tablefmt="pretty"))
    return test_results


def calculate_preference_probabilities(data):
    """
    Calculate the probabilities of having a parking preference and the conditional probabilities of preferring right or left.

    Parameters:
    - data (pd.DataFrame): DataFrame containing parking preferences.

    Returns:
    - probabilities (dict): Dictionary containing calculated probabilities.
    """
    preferences = data["Parking Preference"]
    total_len = len(preferences)
    has_preference = preferences[(preferences == "Right") | (preferences == "Left")]

    probabilities = {
        "probability_of_preference": len(has_preference) / total_len,
        "probability_of_no_preference": 1 - len(has_preference) / total_len,
        "probability_of_right": has_preference[has_preference == "Right"].count() / len(has_preference),
        "probability_of_left": has_preference[has_preference == "Left"].count() / len(has_preference),
        "count_total": total_len,
        "count_preference": len(has_preference),
        "count_right": has_preference[has_preference == "Right"].count(),
        "count_left": has_preference[has_preference == "Left"].count()

    }

    return probabilities


def analyze_distributions(data, list_of_attributes):
    """
    Analyze the distributions of the given attributes in the data.
    :param data:
    :param list_of_attributes:
    :return:
    """
    results = []

    output_parameters = {}

    # when calculating distribution of shop time we remove non zeroes
    for attribute in list_of_attributes:
        if attribute == "Shop time":
            data = data[data[attribute] > 0]

        output_parameters[attribute] = {}

        # m = np.mean(data[attribute])
        # fit_uniform_dist = stats.uniform(loc=0, scale=2 * m)

        fit_uniform_dist = stats.uniform(loc=(data[attribute].min()),
                                         scale=data[attribute].max() - data[attribute].min())

        test = stats.kstest(data[attribute], fit_uniform_dist.cdf)

        p_value = test[1]
        fit_status = "Good fit" if p_value > 0.05 else "Bad fit"
        results.append([attribute, "Uniform Distribution", fit_status, p_value])
        output_parameters[attribute]["Uniform Distribution"] = ({
            "loc": data[attribute].min(),
            "scale": data[attribute].max() - data[attribute].min()
        })

        # Exponential distribution
        m1 = np.mean(data[attribute])
        fit_exponential_dist = stats.expon(scale=1 / m1)
        test = stats.kstest(data[attribute], fit_exponential_dist.cdf)
        p_value = test[1]
        fit_status = "Good fit" if p_value > 0.05 else "Bad fit"
        results.append([attribute, "Exponential Distribution", fit_status, p_value])
        output_parameters[attribute]["Exponential Distribution"] = ({
            "estimated_lambda": 1 / m1
        })

        # Gamma distribution
        m2 = np.mean([x ** 2 for x in data[attribute]])
        est_beta = m1 / (m2 - m1 ** 2)
        est_alpha = m1 * est_beta
        fit_gamma_dist = stats.gamma(a=est_alpha, scale=1 / est_beta)
        test = stats.kstest(data[attribute], fit_gamma_dist.cdf)
        p_value = test[1]
        fit_status = "Good fit" if p_value > 0.05 else "Bad fit"
        results.append([attribute, "Gamma Distribution", fit_status, p_value])
        output_parameters[attribute]["Gamma Distribution"] = ({
            "estimated_alpha": est_alpha,
            "estimated_beta": est_beta
        })

        # Poisson distribution
        fi_poisson_dist = stats.poisson(mu=m1)
        test = stats.kstest(data[attribute], fi_poisson_dist.cdf)
        p_value = test[1]
        fit_status = "Good fit" if p_value > 0.05 else "Bad fit"
        results.append([attribute, "Poisson Distribution", fit_status, p_value])

        output_parameters[attribute]["Poisson Distribution"] = ({
            "estimated_lambda": m1
        })

        # Normal distribution
        estimated_std = m2 - m1 ** 2
        fit_normal_dist = stats.norm(loc=m1, scale=estimated_std)
        test = stats.kstest(data[attribute], fit_normal_dist.cdf)
        p_value = test[1]
        fit_status = "Good fit" if p_value > 0.05 else "Bad fit"
        results.append([attribute, "Normal Distribution", fit_status, np.format_float_scientific(test[1], precision=2)
                        ])

        output_parameters[attribute]["Normal Distribution"] = ({
            "estimated_mean": m1,
            "estimated_std": estimated_std
        })

    # Convert the results list to a DataFrame for easier manipulation
    results_df = pd.DataFrame(results, columns=["Attribute", "Distribution", "Fit Status", "P-value"])

    return results_df, output_parameters


def selecting_distribution(fit_status, parameters):
    """
    Select the best distribution for each attribute.

    Parameters:
    - fit_status (pd.DataFrame): DataFrame containing the results of the distribution analysis.
    - parameters (dict): Dictionary containing the parameters of the distributions.

    Returns:
    - selected_distributions (dict): Dictionary containing the selected distributions for each attribute.
    """
    selected_distributions = {}

    # Filter the parameters DataFrame to only include good fits
    good_fits = fit_status[fit_status["Fit Status"] == "Good fit"]

    # Create a dictionary to store the distribution type and p-value for each attribute
    dist_type = {}

    # Iterate over the good fits
    for index, row in good_fits.iterrows():
        # If the attribute is not in the dictionary, add it
        if row["Attribute"] not in dist_type:
            dist_type[row["Attribute"]] = (row["Distribution"], row["P-value"])
        # If the p-value is higher than the current p-value, update the distribution type
        else:
            if row["P-value"] > dist_type[row["Attribute"]][1]:
                dist_type[row["Attribute"]] = (row["Distribution"], row["P-value"])

    # Create a dictionary to store the parameters of the selected distributions
    results = {}

    # Iterate over the distribution types
    for attribute, value in dist_type.items():
        # Add the distribution type and parameters to the results dictionary
        results[attribute] = (value[0], parameters[attribute][value[0]])

    # Return the selected distributions
    return results


def main():
    """
    Main function to execute the Exploratory Data Analysis (EDA) process.

    The function performs the following steps:
    1. Load the data using the `data_loading` function.
    2. Calculate the statistics of the data using the `calculate_statistics` function.
    3. Calculate the probabilities of shop time using the `calculate_shop_probabilities` function.
    4. Calculate the probabilities of parking preferences using the `calculate_preference_probabilities` function.
    5. Perform a binomial test on the parking preferences probabilities using the `test_binomial_distribution` function.
    6. Analyze the distributions of the data using the `analyze_distributions` function.
    7. Select the best distribution for each attribute using the `selecting_distribution` function.
    8. Plot the distribution of the service time fuel.
    9. Plot the distribution of the service time payment.
    10. Plot the distribution of the shop time.
    11. Plot the distribution of the interarrival times.
    12. Plot a histogram of the frequency of arrival times in the day.

    The function saves the plots in the `graphs/EDA` directory.
    """

    data = data_loading()
    statistics = calculate_statistics(
        data,
        ['Service time Fuel', 'Service time payment', 'Interarrival Times', 'Shop time']
    )

    shop_probabilities = calculate_shop_probabilities(data)

    preferences_probabilities = calculate_preference_probabilities(data)
    test_bin = test_binomial_distribution(preferences_probabilities)

    list_of_attributes = ["Service time Fuel", "Service time payment", "Interarrival Times", "Shop time"]
    fit_status, parameters = analyze_distributions(data, list_of_attributes)

    selected_distributions = selecting_distribution(fit_status, parameters)

    # plot the distribution of the service time fuel
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(data["Service time Fuel"], density=True)
    ax.set_xlabel("Service Time Fuel (s)")
    ax.set_ylabel("Density")
    x = np.linspace(data["Service time Fuel"].min(), data["Service time Fuel"].max())
    y = stats.gamma(a=parameters["Service time Fuel"]["Gamma Distribution"]["estimated_alpha"],
                    scale=1 / parameters["Service time Fuel"]["Gamma Distribution"]["estimated_beta"]).pdf(x)
    ax.plot(
        x,
        y,
        label="Estimated Gamma Distribution",
        color="orange"
    )
    ax.legend(loc="upper right")
    fig.suptitle("Service Time Fuel Distribution")
    fig.savefig("graphs/EDA/service_time_fuel_distribution.png")

    # plot the distribution of the service time payment
    fig, ax = plt.subplots(figsize=(10, 5))
    observed_frequencies = data['Service time payment'].value_counts().sort_index()

    # λ for the Poisson distribution from the data
    lambda_poisson = data['Service time payment'].mean()

    # Re-create the Poisson distribution object
    poisson_dist = stats.poisson(mu=lambda_poisson)

    # Recalculate the expected Poisson frequencies for each unique value
    expected_frequencies_poisson = poisson_dist.pmf(observed_frequencies.index) * len(data)

    ax.bar(observed_frequencies.index, observed_frequencies, label='Observed Distribution')

    # Add a scatter plot for the expected Poisson frequencies
    ax.scatter(observed_frequencies.index, expected_frequencies_poisson, color='orange',
               label='Expected Poisson Distribution', zorder=2)

    ax.set_xlabel("Service Time Payment (s)")
    ax.set_ylabel("Count (Customers)")

    ax.legend(loc="upper right")

    fig.suptitle("Service Time Payment Distribution")
    fig.savefig("graphs/EDA/service_time_payment_distribution.png")

    # plot the distribution of the interarrival times
    fig, ax = plt.subplots(figsize=(10, 5))
    shop_data = data[data["Shop time"] > 0]

    ax.hist(shop_data["Shop time"], density=True)
    ax.set_xlabel("Shop Time (s)")
    ax.set_ylabel("Density")
    # plot the gamma distribution
    x = np.linspace(shop_data["Shop time"].min(), shop_data["Shop time"].max())
    y = stats.gamma(a=parameters["Shop time"]["Gamma Distribution"]["estimated_alpha"],
                    scale=1 / parameters["Shop time"]["Gamma Distribution"]["estimated_beta"]).pdf(x)

    ax.plot(
        x,
        y,
        label="Estimated Gamma Distribution",
        color="orange"
    )
    ax.legend(loc="upper right")

    fig.suptitle("Shop Time Distribution")
    fig.savefig("graphs/EDA/shop_time_distribution.png")

    # plot the distribution of the interarrival times
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(data["Interarrival Times"], density=True)
    ax.set_xlabel("Interarrival Times (s)")
    ax.set_ylabel("Density")

    x = np.linspace(data["Interarrival Times"].min(), data["Interarrival Times"].max())
    y = stats.gamma(a=parameters["Interarrival Times"]["Gamma Distribution"]["estimated_alpha"],
                    scale=1 / parameters["Interarrival Times"]["Gamma Distribution"]["estimated_beta"]).pdf(x)

    ax.plot(
        x,
        y,
        label="Estimated Gamma Distribution",
        color="orange"
    )
    ax.legend(loc="upper right")

    fig.suptitle("Interarrival Times Distribution")
    fig.savefig("graphs/EDA/interarrival_times_distribution.png")

    # plot a histogram of frequency of arrival times in the day
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(data["Arrival Time (H)"], bins=16, edgecolor="black", density=True)

    ax.set_xlabel("Hour of the Day")
    ax.set_ylabel("Density")
    fig.suptitle("Distribution of Arrivals in the Day")
    fig.savefig("graphs/EDA/arrival_time_distribution.png")
