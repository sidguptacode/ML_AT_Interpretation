from __future__ import print_function
import numpy as np
import sys
sys.path.insert(0,'..')
from spreadsheet_editing.spreadsheet_editing import read_spreadsheet, get_spreadsheet_service
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score


def compute_confusion_matrix(p1, p2, urls_labelled_by_both):
    """Computes a confusion matrix using numpy for two np.arrays.
    Results are identical (and similar in computation time) to: 
    "from sklearn.metrics import confusion_matrix"
    However, this function avoids the dependency on sklearn."""

    # K = max(len(np.unique(p1)), len(np.unique(p2))) # Number of classes 
    K = 5
    result = np.zeros((K, K))
    disagreement_url_lst = []
    one_off_by_one_url_lst = []
    urls_with_one = []
    for i in range(len(p1)):
        p1_label, p2_label = int(p1[i]), int(p2[i])
        if np.abs(p1_label - p2_label) >= 2:
            disagreement_url_lst.append(f"{urls_labelled_by_both[i]} {p1_label} {p2_label}")
        if p1_label == 1 or p2_label == 1:
            urls_with_one.append(f"{urls_labelled_by_both[i]} {p1_label} {p2_label}")
        if p1_label == 1 or p2_label == 1 and np.abs(p1_label - p2_label) >= 1:
            one_off_by_one_url_lst.append(f"{urls_labelled_by_both[i]} {p1_label} {p2_label}")
        result[p1_label][p2_label] += 1

    return result, disagreement_url_lst, one_off_by_one_url_lst, urls_with_one


def get_kappas_score(p1, p2):
    p1, p2 = [int(p1_label) for p1_label in p1], [int(p2_label) for p2_label in p2]
    k_s = cohen_kappa_score(p1, p2)
    return k_s


def create_confusion_matrices(labeller_matrix, labeller_emails, url_list):
    # For each pair of labellers, create a confusion matrix to see if someone is under-labelling, over-labelling, etc
    num_labellers = labeller_matrix.shape[1]
    conf_matrices = []
    disagreement_urls = {}
    one_off_by_one_urls = {}
    total_one_off_by_one_urls = []
    total_disagreed_urls = []
    total_labelled_urls = []
    total_urls_with_one = []
    for p1_ind in range(num_labellers):
        for p2_ind in range(p1_ind + 1, num_labellers):
            p1_col, p2_col = labeller_matrix[:, p1_ind], labeller_matrix[:, p2_ind]
            p1_col, p2_col = np.expand_dims(p1_col, -1), np.expand_dims(p2_col, -1)
            comparison_matrix = np.concatenate((p1_col, p2_col), axis=1)
            # Filter out all rows that contain a -1 (unlabelled by one labeller)
            unlabelled_inds = np.where(comparison_matrix < 0)[0]
            unlabelled_inds = np.unique(unlabelled_inds)
            comparison_matrix = np.delete(comparison_matrix, unlabelled_inds, axis=0)
            urls_labelled_by_both = np.delete(url_list, unlabelled_inds, axis=0)
            p1_col, p2_col = comparison_matrix[:, 0], comparison_matrix[:, 1]
            conf_matrix, disagreement_url_lst, one_off_by_one_url_lst, urls_with_one = compute_confusion_matrix(p1_col, p2_col, urls_labelled_by_both)
            kappa_coef = get_kappas_score(p1_col, p2_col)
            # conf_matrix = confusion_matrix(comparison_matrix[:, 0], comparison_matrix[:, 1])
            conf_matrices.append((conf_matrix, p1_ind, p2_ind, kappa_coef))
            disagreement_urls[p1_ind, p2_ind] = disagreement_url_lst
            one_off_by_one_urls[p1_ind, p2_ind] = one_off_by_one_url_lst
            total_one_off_by_one_urls.extend(one_off_by_one_url_lst)
            total_disagreed_urls.extend(disagreement_url_lst)
            total_labelled_urls.extend(urls_labelled_by_both)
            total_urls_with_one.extend(urls_with_one)

    avg_kappas_coef, coef_count = 0, 0
    with open(f"./labels_info.txt", "w") as outf:
        for conf_matrix, p1_ind, p2_ind, kappa_coef in conf_matrices:
            if 'sidgupt1999@gmail.com' in labeller_emails[p1_ind] or 'sidgupt1999@gmail.com' in labeller_emails[p2_ind]:
                continue
            outf.write('==========================\n')
            outf.write(f"Confusion matrix for rows={labeller_emails[p1_ind]} cols={labeller_emails[p2_ind]}\n")
            outf.write(f'{conf_matrix}\n')
            outf.write(f'Kappa\'s coefficient: {kappa_coef}\n')
            avg_kappas_coef += kappa_coef
            coef_count += 1
            outf.write(f'==========================\n')
    avg_kappas_coef /= coef_count

    with open(f"./labels_info.txt", "a") as outf:
        outf.write(f'==========================\n')
        outf.write(f'Average Kappa\'s coefficient: {avg_kappas_coef}\n')
        outf.write(f'==========================\n')

    with open(f"./labels_info.txt", "a") as outf:
        for p1_ind, p2_ind in disagreement_urls:
            outf.write('==========================\n')
            outf.write(f"Off-by-2 disagreement URLS for {labeller_emails[p1_ind]} and {labeller_emails[p2_ind]}\n")
            disagreement_str = "\n".join(disagreement_urls[p1_ind, p2_ind])
            outf.write(f'{disagreement_str}\n')
            outf.write(f'==========================\n')
    # with open(f"./labels_info.txt", "a") as outf:
    #     for p1_ind, p2_ind in one_off_by_one_urls:
    #         outf.write('==========================\n')
    #         outf.write(f"Off by one (for label=1) URLS for {labeller_emails[p1_ind]} and {labeller_emails[p2_ind]}\n")
    #         one_off_by_one_str = "\n".join(one_off_by_one_urls[p1_ind, p2_ind])
    #         outf.write(f'{one_off_by_one_str}\n')
    #         outf.write(f'==========================\n')
    total_disagreed_count, total_labelled_count = len(np.unique(np.array(total_disagreed_urls))), len(np.unique(np.array(total_labelled_urls)))
    with open(f"./labels_info.txt", "a") as outf:
        outf.write('==========================\n')
        outf.write(f"Total disagreed URLs: {total_disagreed_count} ({int(100 * total_disagreed_count / total_labelled_count)} % of total urls labelled by >=2 people)\n")
        outf.write(f'==========================\n')
    total_one_off_by_one_count, total_labelled_count = len(np.unique(np.array(total_one_off_by_one_urls))), len(np.unique(np.array(total_urls_with_one)))
    # with open(f"./labels_info.txt", "a") as outf:
    #     outf.write('==========================\n')
    #     outf.write(f"Total one-off-by-one URLs: {total_one_off_by_one_count} ({int(100 * total_one_off_by_one_count / total_labelled_count)} % of total urls labelled by >=2 people)\n")
    #     outf.write(f'==========================\n')


def output_label_counts(label_counts, type='uni'):
    with open(f"./labels_info.txt", "a") as outf:
        outf.write(f'For {type} labels:\n')
        for i in range(len(label_counts)):
            outf.write(f'{i} - {label_counts[i]} occurances ({int(100 * label_counts[i] / sum(label_counts))} % of the dataset)\n')
        outf.write(f'{sum(label_counts)} labels in total\n')


def output_url_list(url_strs, url_list_type):
    with open(f"./labels_info.txt", "a") as outf:
        outf.write('==========================\n')
        outf.write(f'{url_list_type} URLs:\n')
        if url_list_type == 'Unanimous':
            outf.write(f'{url_strs}\n')
        else:
            for url_str in url_strs:
                outf.write(f'{url_str}\n')


def _has_off_by_one_error(labels):
    num_labels = len(labels)
    if num_labels == 1:
        return False
    for i in range(num_labels):
        for j in range(i, num_labels):
            if np.abs(labels[i] - labels[j]) >= 1:
                return True
    return False


def contains_valid_tray(url, trays):
    for tray in trays:
        if tray in url:
            return True
    return False


def pull_labels(print_confusion_matrices):
    service = get_spreadsheet_service(token_path='../spreadsheet_editing/token.json', creds_path='../spreadsheet_editing/credentials.json')
    spreadsheet_cols = read_spreadsheet(service)
    spreadsheet_rows = np.array(spreadsheet_cols).T
    labeller_emails = spreadsheet_rows[0, 1:]
    spreadsheet_rows = spreadsheet_rows[1:]
    num_urls = len(spreadsheet_rows)
    num_labellers = len(labeller_emails)
    labeller_matrix = np.ones((num_urls, num_labellers)) * -1

    valid_uni_trays = ["20210708_14_38", "20210708_13_33", "20210727_15_02", "20210727_14_54"]

    uni_labels_dict = {}
    multi_labels_dict = {}
    uni_labels_dict_all = {}
    multi_labels_dict_all = {}

    # Used to count the distribution of labels
    uni_label_counts = np.zeros(5)
    multi_label_counts = np.zeros(5)
    uncertain_url_strs = []
    unanimous_url_strs = {0: [], 1: [], 2: [], 3: [], 4: []}

    for i, row in enumerate(spreadsheet_rows):
        uncertain_url = False
        url, labels = row[0], row[1:]
        if "N/A" in labels or all(l == "-1" for l in labels):
            # We won't have N/A or unlabelled wells in our dataset
            continue
        labels = list(map(int, labels))
        labeller_matrix[i, :] = labels[:]
        # Filter out the -1 and -2 labels
        valid_labels = list(filter(lambda l: l != -1, labels))
        if -2 in labels:
            uncertain_url = True
        valid_labels = list(filter(lambda l: l != -2, valid_labels))
        if uncertain_url and _has_off_by_one_error(valid_labels):
            uncertain_url_strs.append(f"{url} | {valid_labels} | {np.mean(valid_labels)}")
        if len(valid_labels) >= 1 and len(np.unique(valid_labels)) == 1 and valid_labels[0] == 4:
            label = int(valid_labels[0])
            unanimous_url_strs[label].append(f"{url}")
            # unanimous_url_strs.append(f"{url} | {valid_labels} | {np.mean(valid_labels)}")

        if len(valid_labels) == 1:
            if contains_valid_tray(url, valid_uni_trays):
                labels_dict = uni_labels_dict
                labels_dict_all = uni_labels_dict_all
                label_counts = uni_label_counts
            else:
                continue
            #     if valid_labels[0] == 0:
            #         continue
            #     if valid_labels[0] == 4 and "rabbit" not in url:
            #         continue
            #     labels_dict = uni_labels_dict
            #     labels_dict_all = uni_labels_dict_all
            #     label_counts = uni_label_counts
        else: 
            labels_dict = multi_labels_dict
            labels_dict_all = multi_labels_dict_all
            label_counts = multi_label_counts

        avged_label = np.mean(np.array(valid_labels))
        label_counts[int(np.around(avged_label))] += 1

        # Fill in the labels_dict
        # an example img_name: 20210624_16_11_B3_10
        img_name = url.split('/')[-1].split('.png')[0]
        tray_name = '_'.join(img_name.split('_')[:-2])
        well_coord, frame = img_name.split('_')[-2:]
        if tray_name not in labels_dict:
            labels_dict[tray_name] = {}
            labels_dict_all[tray_name] = {}
        labels_dict[tray_name][well_coord, frame] = avged_label
        labels_dict_all[tray_name][well_coord, frame] = np.array(valid_labels)

    if print_confusion_matrices:
        create_confusion_matrices(labeller_matrix, labeller_emails, spreadsheet_rows[:, 0])

    # Print the distribution of labels
    output_label_counts(uni_label_counts, type='uni')
    output_label_counts(multi_label_counts, type='multi')

    # Print the uncertain URLs and their labels
    output_url_list(uncertain_url_strs, 'Uncertain')
    # output_url_list(unanimous_url_strs, 'Unanimous')

    return uni_labels_dict, multi_labels_dict, multi_labels_dict_all

if __name__ == '__main__':
    pull_labels(True)