# monday_extract_groups.py

import os
import requests
import sys
import csv
from tqdm import tqdm
import pandas as pd
import re
import warnings
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def fetch_groups(board_id, api_key):
    """
    Fetches groups from a specified Monday.com board.

    Args:
        board_id (str): The ID of the board.
        api_key (str): Your Monday.com API key.

    Returns:
        list: A list of groups with their IDs and titles.
    """
    query = """
    query ($boardId: [ID!]!) {
      boards(ids: $boardId) {
        groups {
          id
          title
        }
      }
    }
    """

    variables = {"boardId": [str(board_id)]}

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    response = requests.post("https://api.monday.com/v2",
                             json={
                                 "query": query,
                                 "variables": variables
                             },
                             headers=headers)

    if response.status_code != 200:
        raise Exception(
            f"Query failed with status code {response.status_code}: {response.text}"
        )

    data = response.json()

    if 'errors' in data:
        error_messages = "\n".join(
            [error['message'] for error in data['errors']])
        raise Exception(f"GraphQL Errors:\n{error_messages}")

    boards = data.get('data', {}).get('boards', [])
    if not boards:
        raise Exception(f"No boards found with ID {board_id}.")

    board = boards[0]
    groups = board.get('groups', [])

    if not groups:
        raise Exception(f"No groups found in board {board_id}.")

    return groups


def fetch_items(board_id, group_id, api_key, limit=10):
    """
    Fetches items from a specific group within a Monday.com board.

    Args:
        board_id (str): The ID of the board.
        group_id (str): The ID of the group.
        api_key (str): Your Monday.com API key.
        limit (int): Number of items to fetch.

    Returns:
        list: A list of items with their details.
    """
    query = """
    query ($boardId: [ID!]!, $groupId: [String!]!, $limit: Int!) {
      boards(ids: $boardId) {
        groups(ids: $groupId) {
          id
          title
          items_page(limit: $limit) {
            items {
              id
              name
              column_values {
                id
                text
              }
            }
          }
        }
      }
    }
    """

    variables = {
        "boardId": [str(board_id)],
        "groupId": [str(group_id)],
        "limit": limit
    }

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    response = requests.post("https://api.monday.com/v2",
                             json={
                                 "query": query,
                                 "variables": variables
                             },
                             headers=headers)

    if response.status_code != 200:
        raise Exception(
            f"Query failed with status code {response.status_code}: {response.text}"
        )

    data = response.json()

    if 'errors' in data:
        error_messages = "\n".join(
            [error['message'] for error in data['errors']])
        raise Exception(f"GraphQL Errors:\n{error_messages}")

    boards = data.get('data', {}).get('boards', [])
    if not boards:
        raise Exception(f"No boards found with ID {board_id}.")

    board = boards[0]
    groups = board.get('groups', [])
    if not groups:
        raise Exception(
            f"No groups found with ID '{group_id}' in board {board_id}.")

    group = groups[0]
    items_page = group.get('items_page', {})
    items = items_page.get('items', [])

    return items


def export_items_to_csv(items, filename):
    """
    Exports fetched items to a CSV file.

    Args:
        items (list): List of items to export.
        filename (str): The name of the CSV file.
    """
    if not items:
        return

    headers = ['Item ID', 'Item Name']
    column_ids = []
    for column in items[0]['column_values']:
        headers.append(column['id'])
        column_ids.append(column['id'])

    with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        for item in items:
            row = {'Item ID': item['id'], 'Item Name': item['name']}
            for column in item['column_values']:
                row[column['id']] = column.get('text', '')
            writer.writerow(row)


def fetch_items_recursive(board_id, group_id, api_key, limit=500):
    """
    Recursively fetches all items from a specific group within a Monday.com board using cursor-based pagination.

    Args:
        board_id (str): The ID of the board.
        group_id (str): The ID of the group.
        api_key (str): Your Monday.com API key.
        limit (int, optional): Number of items to fetch per request. Defaults to 500.

    Returns:
        list: A complete list of all items in the group.
    """
    all_items = []
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    initial_query = """
    query ($boardId: [ID!]!, $groupId: [String!]!, $limit: Int!) {
      boards(ids: $boardId) {
        groups(ids: $groupId) {
          id
          title
          items_page(limit: $limit) {
            cursor
            items {
              id
              name
              column_values {
                id
                text
              }
            }
          }
        }
      }
    }
    """

    variables = {
        "boardId": [str(board_id)],
        "groupId": [str(group_id)],
        "limit": limit
    }

    response = requests.post("https://api.monday.com/v2",
                             json={
                                 "query": initial_query,
                                 "variables": variables
                             },
                             headers=headers)

    if response.status_code != 200:
        raise Exception(
            f"Initial query failed with status code {response.status_code}: {response.text}"
        )

    data = response.json()

    if 'errors' in data:
        error_messages = "\n".join(
            [error['message'] for error in data['errors']])
        raise Exception(f"GraphQL Errors in initial query:\n{error_messages}")

    try:
        group = data['data']['boards'][0]['groups'][0]
        items_page = group.get('items_page', {})
        items = items_page.get('items', [])
        all_items.extend(items)
        cursor = items_page.get('cursor')
    except (IndexError, KeyError) as e:
        raise Exception(f"Error parsing initial response: {e}")

    while cursor:
        next_query = """
        query ($limit: Int!, $cursor: String!) {
          next_items_page(limit: $limit, cursor: $cursor) {
            cursor
            items {
              id
              name
              column_values {
                id
                text
              }
            }
          }
        }
        """

        next_variables = {"limit": limit, "cursor": cursor}

        response = requests.post("https://api.monday.com/v2",
                                 json={
                                     "query": next_query,
                                     "variables": next_variables
                                 },
                                 headers=headers)

        if response.status_code != 200:
            raise Exception(
                f"Next items query failed with status code {response.status_code}: {response.text}"
            )

        data = response.json()

        if 'errors' in data:
            error_messages = "\n".join(
                [error['message'] for error in data['errors']])
            raise Exception(
                f"GraphQL Errors in next_items_page query:\n{error_messages}")

        try:
            next_page = data['data']['next_items_page']
            items = next_page.get('items', [])
            all_items.extend(items)
            cursor = next_page.get('cursor')
        except (KeyError, TypeError) as e:
            raise Exception(f"Error parsing next_items_page response: {e}")

    return all_items


def fetch_and_export_all_groups(board_id,
                                group_list,
                                name_list,
                                api_key,
                                limit=500):
    """
    Fetches items from all specified groups and exports them to corresponding CSV files.

    Args:
        board_id (str): The ID of the board.
        group_list (list): List of group IDs to fetch.
        name_list (list): List of filenames for each group.
        api_key (str): Your Monday.com API key.
        limit (int, optional): Number of items to fetch per request. Defaults to 500.
    """
    groups = fetch_groups(board_id, api_key)
    group_dict = {group['id']: group for group in groups}

    for group_id, filename in tqdm(zip(group_list, name_list),
                                   total=len(group_list),
                                   desc="Fetching Groups"):
        if group_id not in group_dict:
            # Optionally, handle missing groups as needed
            continue

        items = fetch_items_recursive(board_id, group_id, api_key, limit)
        export_items_to_csv(items, filename)


date_pattern = r"\d{4}-\d{2}-\d{2}"


def extract_date(value):
    """
    Extracts date from a string using a regex pattern.
    """
    if pd.isna(value) or value == 'NaT':
        return None  # Handle NaT or NaN values
    if isinstance(value, str):
        match = re.search(date_pattern, value)
        return match.group(0) if match else None
    return None


def items_to_dataframe(items):
    """
    Converts a list of items to a pandas DataFrame.
    """
    if not items:
        logger.warning("No items to convert.")
        return pd.DataFrame()

    data = []
    column_ids = [column['id'] for column in items[0]['column_values']]
    headers = ['Item ID', 'Item Name'] + column_ids

    for item in items:
        row = {'Item ID': item['id'], 'Item Name': item['name']}
        for column in item['column_values']:
            row[column['id']] = column.get('text', '')
        data.append(row)

    df = pd.DataFrame(data, columns=headers)
    return df


def fetch_data():
    """
    Fetches data from Monday.com and returns a dictionary of DataFrames.
    """
    BOARD_ID = "6942829967"
    group_list = [
        "topics", "new_group34578__1", "new_group27351__1",
        "new_group54376__1", "new_group64021__1", "new_group65903__1",
        "new_group62617__1"
    ]
    name_list = [
        "scheduled", "unqualified", "won", "cancelled", "noshow", "proposal",
        "lost"
    ]
    LIMIT = 500  # Items limit per group

    # Fetch API key from secrets
    try:
        api_key = os.getenv("MONDAY_API_KEY")
    except KeyError:
        logger.error("Error: MONDAY_API_KEY is not set in .env.")

    # Fetch all groups from the board
    try:
        api_key = os.getenv("MONDAY_API_KEY")
        groups = fetch_groups(BOARD_ID, api_key)
    except Exception as e:
        logger.error(f"Error fetching groups: {e}")

    dataframes = {}

    total_groups = len(group_list)
    progress_percentage = 0.0
    progress_step = 100 / total_groups

    for i, (group_id, group_name) in tqdm(enumerate(zip(group_list,
                                                        name_list))):
        # Find the target group
        target_group = next(
            (group for group in groups if group['id'] == group_id), None)
        if not target_group:
            logger.error(
                f"Group with ID '{group_id}' not found in board {BOARD_ID}.")

        print(
            f"Fetching items from Group: **{target_group['title']}** (ID: {target_group['id']})"
        )

        # Fetch items from the target group
        try:
            items = fetch_items_recursive(BOARD_ID, target_group['id'],
                                          api_key, LIMIT)
        except Exception as e:
            logger.error(f"Error fetching items for group '{group_name}': {e}")

        df_items = items_to_dataframe(items)
        dataframes[group_name] = df_items

        # Update progress percentage
        progress_percentage += progress_step
        print(f"Progress: {progress_percentage:.2f}%")

    # Define column renaming mapping
    columns_with_titles = {
        'name': 'Name',
        'auto_number__1': 'Auto number',
        'person': 'Owner',
        'last_updated__1': 'Last updated',
        'link__1': 'Linkedin',
        'phone__1': 'Phone',
        'email__1': 'Email',
        'text7__1': 'Company',
        'date4': 'Sales Call Date',
        'status9__1': 'Follow Up Tracker',
        'notes__1': 'Notes',
        'interested_in__1': 'Interested In',
        'status4__1': 'Plan Type',
        'numbers__1': 'Deal Value',
        'status6__1': 'Email Template #1',
        'dup__of_email_template__1': 'Email Template #2',
        'status__1': 'Deal Status',
        'status2__1': 'Send Panda Doc?',
        'utm_source__1': 'UTM Source',
        'date__1': 'Deal Status Date',
        'utm_campaign__1': 'UTM Campaign',
        'utm_medium__1': 'UTM Medium',
        'utm_content__1': 'UTM Content',
        'link3__1': 'UTM LINK',
        'lead_source8__1': 'Lead Source',
        'color__1': 'Channel FOR FUNNEL METRICS',
        'subitems__1': 'Subitems',
        'date5__1': 'Date Created'
    }

    # Rename columns in each dataframe
    for key in dataframes.keys():
        df = dataframes[key]
        df.rename(columns=columns_with_titles, inplace=True)
        dataframes[key] = df

    return dataframes


# ──────────────────────────────────────────────────────────────────────────────
#  Re-implemented process_data – production-ready, no NaN/None in rate columns
# ──────────────────────────────────────────────────────────────────────────────
# def process_data(dataframes: dict[str, pd.DataFrame], st_date: str,
#                  end_date: str, filter_column: str) -> pd.DataFrame:
#     """
#     Build the KPI table for the date range [st_date, end_date] (inclusive).

#     Parameters
#     ----------
#     dataframes     output of fetch_data(); keys such as 'scheduled', 'won', …
#     st_date        'YYYY-MM-DD' – range start
#     end_date       'YYYY-MM-DD' – range end
#     filter_column  column chosen in the UI for date filtering
#                    (usually 'Date Created' , 'Sales Call Date' , '')
#     """
#     # ── unpack individual stages (empty DF if missing) ────────────────────
#     op_cancelled = dataframes.get("cancelled", pd.DataFrame())
#     op_lost = dataframes.get("lost", pd.DataFrame())
#     op_noshow = dataframes.get("noshow", pd.DataFrame())
#     op_proposal = dataframes.get("proposal", pd.DataFrame())
#     op_scheduled = dataframes.get("scheduled", pd.DataFrame())
#     op_unqualified = dataframes.get("unqualified", pd.DataFrame())
#     op_won = dataframes.get("won", pd.DataFrame())

#     # ── canonical list of owners  (whitespace already normalised in fetch_data)
#     owners = (pd.concat([
#         op_cancelled, op_lost, op_noshow, op_proposal, op_scheduled,
#         op_unqualified, op_won
#     ])["Owner"].dropna().unique())
#     kpi = pd.DataFrame(index=owners)
#     kpi.index.name = "Owner"
#     kpi["Owner"] = kpi.index  # explicit column for display

#     # ── convenience: date-range filter  ───────────────────────────────────
#     def _filter(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
#         if date_col not in df.columns:
#             return pd.DataFrame(columns=df.columns)

#         dates = pd.to_datetime(df[date_col].apply(extract_date),
#                                errors="coerce").dt.date
#         mask = ((dates >= pd.to_datetime(st_date).date()) &
#                 (dates <= pd.to_datetime(end_date).date()))
#         return df.loc[mask]

#     fdate = _filter  # alias

#     # ───────────────────────────────────────────────────────────────────────
#     #  RAW COUNTS
#     # ───────────────────────────────────────────────────────────────────────
#     all_stages = pd.concat([
#         op_cancelled, op_lost, op_noshow, op_proposal, op_scheduled,
#         op_unqualified, op_won
#     ])
#     kpi["New Calls Booked"] = (fdate(
#         all_stages,
#         filter_column).groupby("Owner").size().reindex(kpi.index,
#                                                        fill_value=0))

#     sc_taken_df = pd.concat([op_unqualified, op_proposal, op_won, op_lost])
#     kpi["Sales Call Taken"] = (fdate(
#         sc_taken_df,
#         filter_column).groupby("Owner").size().reindex(kpi.index,
#                                                        fill_value=0))

#     kpi["Unqualified"] = (fdate(op_unqualified,
#                                 filter_column).groupby("Owner").size().reindex(
#                                     kpi.index, fill_value=0))

#     kpi["Cancelled Calls"] = (fdate(
#         op_cancelled,
#         filter_column).groupby("Owner").size().reindex(kpi.index,
#                                                        fill_value=0))

#     # Proposals count = Proposal + Won + Lost (anchor to Sales Call date if present)
#     prop_date_col = ("Sales Call Date" if "Sales Call Date"
#                      in op_proposal.columns else filter_column)
#     kpi["Proposals"] = (pd.concat([op_proposal, op_won, op_lost]).pipe(
#         lambda df: fdate(df, prop_date_col)).groupby("Owner").size().reindex(
#             kpi.index, fill_value=0))

#     # ───────────────────────────────────────────────────────────────────────
#     #  RATE METRICS  (all numeric, no NaN/None)
#     # ───────────────────────────────────────────────────────────────────────
#     with np.errstate(divide="ignore", invalid="ignore"):
#         kpi["Show Rate %"] = (kpi["Sales Call Taken"] / kpi["New Calls Booked"]
#                               ).replace([np.inf, -np.inf], 0).fillna(0) * 100

#         kpi["Unqualified Rate %"] = (kpi["Unqualified"] /
#                                      kpi["New Calls Booked"]).replace(
#                                          [np.inf, -np.inf], 0).fillna(0) * 100

#         kpi["Cancellation Rate %"] = (kpi["Cancelled Calls"] /
#                                       kpi["New Calls Booked"]).replace(
#                                           [np.inf, -np.inf], 0).fillna(0) * 100

#         kpi["Proposal Rate %"] = (kpi["Proposals"] /
#                                   kpi["New Calls Booked"]).replace(
#                                       [np.inf, -np.inf], 0).fillna(0) * 100

#     # Close metrics
#     closes = fdate(op_won, prop_date_col).groupby("Owner").size()
#     kpi["Close"] = closes.reindex(kpi.index, fill_value=0)  # helper column

#     with np.errstate(divide="ignore", invalid="ignore"):
#         kpi["Close Rate %"] = (kpi["Close"] / kpi["New Calls Booked"]).replace(
#             [np.inf, -np.inf], 0).fillna(0) * 100

#         kpi["Close Rate(Show) %"] = (kpi["Close"] /
#                                      kpi["Sales Call Taken"]).replace(
#                                          [np.inf, -np.inf], 0).fillna(0) * 100

#         kpi["Close Rate(MQL) %"] = (kpi["Close"] / kpi["Proposals"].replace(
#             0, np.nan)).replace([np.inf, -np.inf], 0).fillna(0) * 100

#     # ───────────────────────────────────────────────────────────────────────
#     #  REVENUE METRICS
#     # ───────────────────────────────────────────────────────────────────────
#     # Always use the user-selected filter_column for consistency
#     won_rev = fdate(op_won.copy(), filter_column)
#     won_rev["Deal Value"] = pd.to_numeric(won_rev["Deal Value"],
#                                           errors="coerce").fillna(0)

#     rev_sum = won_rev.groupby("Owner")["Deal Value"].sum()
#     kpi["Closed Revenue $"] = rev_sum.reindex(kpi.index, fill_value=0)

#     with np.errstate(divide="ignore", invalid="ignore"):
#         kpi["Revenue Per Call $"] = (kpi["Closed Revenue $"] /
#                                      kpi["New Calls Booked"]).replace(
#                                          [np.inf, -np.inf], 0).fillna(0)

#         kpi["Revenue Per Showed Up $"] = (kpi["Closed Revenue $"] /
#                                           kpi["Sales Call Taken"]).replace(
#                                               [np.inf, -np.inf], 0).fillna(0)

#         kpi["Revenue Per Proposal $"] = (
#             kpi["Closed Revenue $"] /
#             kpi["Proposals"].replace(0, np.nan)).replace([np.inf, -np.inf],
#                                                          0).fillna(0)

#     # Pipeline revenue (open proposals)
#     pipe_rev = fdate(op_proposal.copy(), prop_date_col)
#     pipe_rev["Deal Value"] = pd.to_numeric(pipe_rev["Deal Value"],
#                                            errors="coerce").fillna(0)
#     kpi["Pipeline Revenue $"] = (
#         pipe_rev.groupby("Owner")["Deal Value"].sum().reindex(kpi.index,
#                                                               fill_value=0))

#     # ── TOTAL ROW  ──────────────────────────────────────────────────────────
#     totals = {
#         "Owner": "Total",
#         "New Calls Booked": kpi["New Calls Booked"].sum(),
#         "Sales Call Taken": kpi["Sales Call Taken"].sum(),
#         "Proposals": kpi["Proposals"].sum(),
#         "Show Rate %": kpi["Show Rate %"].mean(),
#         "Unqualified": kpi["Unqualified"].sum(),
#         "Unqualified Rate %": kpi["Unqualified Rate %"].mean(),
#         "Cancelled Calls": kpi["Cancelled Calls"].sum(),
#         "Cancellation Rate %": kpi["Cancellation Rate %"].mean(),
#         "Proposal Rate %": kpi["Proposal Rate %"].mean(),
#         "Close Rate %": kpi["Close Rate %"].mean(),
#         "Close Rate(Show) %": kpi["Close Rate(Show) %"].mean(),
#         "Close Rate(MQL) %": kpi["Close Rate(MQL) %"].mean(),
#         "Closed Revenue $": kpi["Closed Revenue $"].sum(),
#         "Revenue Per Call $": kpi["Revenue Per Call $"].mean(),
#         "Revenue Per Showed Up $": kpi["Revenue Per Showed Up $"].mean(),
#         "Revenue Per Proposal $": kpi["Revenue Per Proposal $"].mean(),
#         "Pipeline Revenue $": kpi["Pipeline Revenue $"].sum(),
#     }

#     kpi_final = (pd.concat([kpi,
#                             pd.DataFrame([totals]).set_index("Owner")
#                             ]).reset_index(drop=True)).drop(columns=["Close"],
#                                                             errors="ignore")

#     return kpi_final


def process_data(dataframes: dict[str, pd.DataFrame], st_date: str,
                 end_date: str, filter_column: str) -> pd.DataFrame:
    """
    Build the KPI table for the date range [st_date, end_date] (inclusive).

    Parameters
    ----------
    dataframes     output of fetch_data(); keys such as 'scheduled', 'won', …
    st_date        'YYYY-MM-DD' – range start
    end_date       'YYYY-MM-DD' – range end
    filter_column  column chosen in the UI for date filtering
           (usually 'Date Created' , 'Sales Call Date' , '')
    """
    # ── unpack individual stages (empty DF if missing) ────────────────────
    op_cancelled = dataframes.get("cancelled", pd.DataFrame())
    op_lost = dataframes.get("lost", pd.DataFrame())
    op_noshow = dataframes.get("noshow", pd.DataFrame())
    op_proposal = dataframes.get("proposal", pd.DataFrame())
    op_scheduled = dataframes.get("scheduled", pd.DataFrame())
    op_unqualified = dataframes.get("unqualified", pd.DataFrame())
    op_won = dataframes.get("won", pd.DataFrame())

    # ── canonical list of owners  (whitespace already normalised in fetch_data)
    owners = (pd.concat([
        op_cancelled, op_lost, op_noshow, op_proposal, op_scheduled,
        op_unqualified, op_won
    ])["Owner"].dropna().unique())
    kpi = pd.DataFrame(index=owners)
    kpi.index.name = "Owner"
    kpi["Owner"] = kpi.index  # explicit column for display

    # ── convenience: date-range filter  ───────────────────────────────────
    def _filter(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        if date_col not in df.columns:
            return pd.DataFrame(columns=df.columns)

        dates = pd.to_datetime(df[date_col].apply(extract_date),
                               errors="coerce").dt.date
        mask = ((dates >= pd.to_datetime(st_date).date()) &
                (dates <= pd.to_datetime(end_date).date()))
        return df.loc[mask]

    fdate = _filter  # alias

    # ───────────────────────────────────────────────────────────────────────
    #  RAW COUNTS
    # ───────────────────────────────────────────────────────────────────────
    all_stages = pd.concat([
        op_cancelled, op_lost, op_noshow, op_proposal, op_scheduled,
        op_unqualified, op_won
    ])
    kpi["New Calls Booked"] = (fdate(
        all_stages,
        filter_column).groupby("Owner").size().reindex(kpi.index,
                                                       fill_value=0))

    sc_taken_df = pd.concat([op_unqualified, op_proposal, op_won, op_lost])
    kpi["Sales Call Taken"] = (fdate(
        sc_taken_df,
        filter_column).groupby("Owner").size().reindex(kpi.index,
                                                       fill_value=0))

    kpi["Unqualified"] = (fdate(op_unqualified,
                                filter_column).groupby("Owner").size().reindex(
                                    kpi.index, fill_value=0))

    kpi["Cancelled Calls"] = (fdate(
        op_cancelled,
        filter_column).groupby("Owner").size().reindex(kpi.index,
                                                       fill_value=0))

    # Proposals count = Proposal + Won + Lost (anchor to Sales Call date if present)
    prop_date_col = ("Sales Call Date" if "Sales Call Date"
                     in op_proposal.columns else filter_column)
    kpi["Proposals"] = (pd.concat([op_proposal, op_won, op_lost]).pipe(
        lambda df: fdate(df, prop_date_col)).groupby("Owner").size().reindex(
            kpi.index, fill_value=0))

    # ───────────────────────────────────────────────────────────────────────
    #  Origin Counts
    # ───────────────────────────────────────────────────────────────────────
    #Count the number of cold emails in origin column and group by owner
    cold_emails = all_stages[all_stages['origin'] == 'cold-email']
    kpi["Cold Emails"] = cold_emails.groupby("Owner").size().reindex(
        kpi.index, fill_value=0)

    #Count the number of linkedin in origin column and group by owner
    # linkedin = all_stages[all_stages['origin'] == 'LinkedIn']
    # kpi["LinkedIn"] = linkedin.groupby("Owner").size().reindex(kpi.index,
    #                                                            fill_value=0)
    kpi["LinkedIn"] = np.where(kpi['New Calls Booked'] > 0,
                               kpi['New Calls Booked'] - kpi['Cold Emails'], 0)

    # ───────────────────────────────────────────────────────────────────────
    #  RATE METRICS  (all numeric, no NaN/None)
    # ───────────────────────────────────────────────────────────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        kpi["Show Rate %"] = (kpi["Sales Call Taken"] / kpi["New Calls Booked"]
                              ).replace([np.inf, -np.inf], 0).fillna(0) * 100

        kpi["Unqualified Rate %"] = (kpi["Unqualified"] /
                                     kpi["New Calls Booked"]).replace(
                                         [np.inf, -np.inf], 0).fillna(0) * 100

        kpi["Cancellation Rate %"] = (kpi["Cancelled Calls"] /
                                      kpi["New Calls Booked"]).replace(
                                          [np.inf, -np.inf], 0).fillna(0) * 100

        kpi["Proposal Rate %"] = (kpi["Proposals"] /
                                  kpi["New Calls Booked"]).replace(
                                      [np.inf, -np.inf], 0).fillna(0) * 100

    # Close metrics
    closes = fdate(op_won, prop_date_col).groupby("Owner").size()
    kpi["Close"] = closes.reindex(kpi.index, fill_value=0)  # helper column

    with np.errstate(divide="ignore", invalid="ignore"):
        kpi["Close Rate %"] = (kpi["Close"] / kpi["New Calls Booked"]).replace(
            [np.inf, -np.inf], 0).fillna(0) * 100

        kpi["Close Rate(Show) %"] = (kpi["Close"] /
                                     kpi["Sales Call Taken"]).replace(
                                         [np.inf, -np.inf], 0).fillna(0) * 100

        kpi["Close Rate(MQL) %"] = (kpi["Close"] / kpi["Proposals"].replace(
            0, np.nan)).replace([np.inf, -np.inf], 0).fillna(0) * 100

    # ───────────────────────────────────────────────────────────────────────
    #  REVENUE METRICS
    # ───────────────────────────────────────────────────────────────────────
    # Always use the user-selected filter_column for consistency
    won_rev = fdate(op_won.copy(), filter_column)
    won_rev["Deal Value"] = pd.to_numeric(won_rev["Deal Value"],
                                          errors="coerce").fillna(0)

    rev_sum = won_rev.groupby("Owner")["Deal Value"].sum()
    kpi["Closed Revenue $"] = rev_sum.reindex(kpi.index, fill_value=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        kpi["Revenue Per Call $"] = (kpi["Closed Revenue $"] /
                                     kpi["New Calls Booked"]).replace(
                                         [np.inf, -np.inf], 0).fillna(0)

        kpi["Revenue Per Showed Up $"] = (kpi["Closed Revenue $"] /
                                          kpi["Sales Call Taken"]).replace(
                                              [np.inf, -np.inf], 0).fillna(0)

        kpi["Revenue Per Proposal $"] = (
            kpi["Closed Revenue $"] /
            kpi["Proposals"].replace(0, np.nan)).replace([np.inf, -np.inf],
                                                         0).fillna(0)

    # Pipeline revenue (open proposals)
    pipe_rev = fdate(op_proposal.copy(), prop_date_col)
    pipe_rev["Deal Value"] = pd.to_numeric(pipe_rev["Deal Value"],
                                           errors="coerce").fillna(0)
    kpi["Pipeline Revenue $"] = (
        pipe_rev.groupby("Owner")["Deal Value"].sum().reindex(kpi.index,
                                                              fill_value=0))

    # ── TOTAL ROW  ────────────────────────────────────────────────────
    totals = {
        "Owner": "Total",
        "New Calls Booked": kpi["New Calls Booked"].sum(),
        "Sales Call Taken": kpi["Sales Call Taken"].sum(),
        "Proposals": kpi["Proposals"].sum(),
        "Show Rate %": kpi["Show Rate %"].mean(),
        "Unqualified": kpi["Unqualified"].sum(),
        "Unqualified Rate %": kpi["Unqualified Rate %"].mean(),
        "Cancelled Calls": kpi["Cancelled Calls"].sum(),
        "Cancellation Rate %": kpi["Cancellation Rate %"].mean(),
        "Proposal Rate %": kpi["Proposal Rate %"].mean(),
        "Close Rate %": kpi["Close Rate %"].mean(),
        "Close Rate(Show) %": kpi["Close Rate(Show) %"].mean(),
        "Close Rate(MQL) %": kpi["Close Rate(MQL) %"].mean(),
        "Closed Revenue $": kpi["Closed Revenue $"].sum(),
        "Revenue Per Call $": kpi["Revenue Per Call $"].mean(),
        "Revenue Per Showed Up $": kpi["Revenue Per Showed Up $"].mean(),
        "Revenue Per Proposal $": kpi["Revenue Per Proposal $"].mean(),
        "Pipeline Revenue $": kpi["Pipeline Revenue $"].sum(),
        "Cold Emails": kpi["Cold Emails"].sum(),
        "LinkedIn": kpi["LinkedIn"].sum(),
    }

    # Create totals row with proper index
    totals_df = pd.DataFrame([totals])

    # Concatenate the main KPI data with totals
    kpi_final = pd.concat([kpi.reset_index(drop=True), totals_df],
                          ignore_index=True).drop(columns=["Close"],
                                                  errors="ignore")
    return kpi_final

def process_data_COLD_EMAIL(dataframes: dict[str, pd.DataFrame], st_date: str,
     end_date: str, filter_column: str) -> pd.DataFrame:
    """
    Build the KPI table for the date range [st_date, end_date] (inclusive).
    
    Parameters
    ----------
    dataframes     output of fetch_data(); keys such as 'scheduled', 'won', …
    st_date        'YYYY-MM-DD' – range start
    end_date       'YYYY-MM-DD' – range end
    filter_column  column chosen in the UI for date filtering
    (usually 'Date Created' , 'Sales Call Date' , '')
    """
    # ── unpack individual stages (empty DF if missing) ────────────────────
    op_cancelled = dataframes.get("cancelled", pd.DataFrame())
    op_lost = dataframes.get("lost", pd.DataFrame())
    op_noshow = dataframes.get("noshow", pd.DataFrame())
    op_proposal = dataframes.get("proposal", pd.DataFrame())
    op_scheduled = dataframes.get("scheduled", pd.DataFrame())
    op_unqualified = dataframes.get("unqualified", pd.DataFrame())
    op_won = dataframes.get("won", pd.DataFrame())
    
    # ── canonical list of owners  (whitespace already normalised in fetch_data)
    owners = (pd.concat([
    op_cancelled, op_lost, op_noshow, op_proposal, op_scheduled,
    op_unqualified, op_won
    ])["Owner"].dropna().unique())
    kpi = pd.DataFrame(index=owners)
    kpi.index.name = "Owner"
    kpi["Owner"] = kpi.index  # explicit column for display
    
    # ── convenience: date-range filter  ───────────────────────────────────
    def _filter(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        if date_col not in df.columns:
            return pd.DataFrame(columns=df.columns)
        
        dates = pd.to_datetime(df[date_col].apply(extract_date),
                           errors="coerce").dt.date
        mask = ((dates >= pd.to_datetime(st_date).date()) &
            (dates <= pd.to_datetime(end_date).date()))
        return df.loc[mask]
    
    fdate = _filter  # alias
    
    # ───────────────────────────────────────────────────────────────────────
    #  RAW COUNTS
    # ───────────────────────────────────────────────────────────────────────
    all_stages = pd.concat([
    op_cancelled, op_lost, op_noshow, op_proposal, op_scheduled,
    op_unqualified, op_won
    ])
    kpi["New Calls Booked"] = (fdate(
    all_stages,
    filter_column).groupby("Owner").size().reindex(kpi.index,
                                               fill_value=0))
    
    sc_taken_df = pd.concat([op_unqualified, op_proposal, op_won, op_lost])
    kpi["Sales Call Taken"] = (fdate(
    sc_taken_df,
    filter_column).groupby("Owner").size().reindex(kpi.index,
                                               fill_value=0))
    
    kpi["Unqualified"] = (fdate(op_unqualified,
                        filter_column).groupby("Owner").size().reindex(
                            kpi.index, fill_value=0))
    
    kpi["Cancelled Calls"] = (fdate(
    op_cancelled,
    filter_column).groupby("Owner").size().reindex(kpi.index,
                                               fill_value=0))
    
    # Proposals count = Proposal + Won + Lost (anchor to Sales Call date if present)
    prop_date_col = ("Sales Call Date" if "Sales Call Date"
             in op_proposal.columns else filter_column)
    kpi["Proposals"] = (pd.concat([op_proposal, op_won, op_lost]).pipe(
    lambda df: fdate(df, prop_date_col)).groupby("Owner").size().reindex(
    kpi.index, fill_value=0))
    
    # ───────────────────────────────────────────────────────────────────────
    #  Origin Counts
    # ───────────────────────────────────────────────────────────────────────
    #Count the number of cold emails in origin column and group by owner
    cold_emails = all_stages[all_stages['origin'] == 'cold-email']
    kpi["Cold Emails"] = cold_emails.groupby("Owner").size().reindex(
    kpi.index, fill_value=0)
    
    #Count the number of linkedin in origin column and group by owner
    # linkedin = all_stages[all_stages['origin'] == 'LinkedIn']
    # kpi["LinkedIn"] = linkedin.groupby("Owner").size().reindex(kpi.index,
    #                                                            fill_value=0)
    # kpi["LinkedIn"] = np.where(kpi['New Calls Booked'] > 0,
    #                    kpi['New Calls Booked'] - kpi['Cold Emails'], 0)
    
    # ───────────────────────────────────────────────────────────────────────
    #  RATE METRICS  (all numeric, no NaN/None)
    # ───────────────────────────────────────────────────────────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        kpi["Show Rate %"] = (kpi["Sales Call Taken"] / kpi["New Calls Booked"]
                          ).replace([np.inf, -np.inf], 0).fillna(0) * 100
        
        kpi["Unqualified Rate %"] = (kpi["Unqualified"] /
                                 kpi["New Calls Booked"]).replace(
                                     [np.inf, -np.inf], 0).fillna(0) * 100
        
        kpi["Cancellation Rate %"] = (kpi["Cancelled Calls"] /
                                  kpi["New Calls Booked"]).replace(
                                      [np.inf, -np.inf], 0).fillna(0) * 100
        
        kpi["Proposal Rate %"] = (kpi["Proposals"] /
                              kpi["New Calls Booked"]).replace(
                                  [np.inf, -np.inf], 0).fillna(0) * 100
    
    # Close metrics
    closes = fdate(op_won, prop_date_col).groupby("Owner").size()
    kpi["Close"] = closes.reindex(kpi.index, fill_value=0)  # helper column
    
    with np.errstate(divide="ignore", invalid="ignore"):
        kpi["Close Rate %"] = (kpi["Close"] / kpi["New Calls Booked"]).replace(
        [np.inf, -np.inf], 0).fillna(0) * 100
        
        kpi["Close Rate(Show) %"] = (kpi["Close"] /
                                 kpi["Sales Call Taken"]).replace(
                                     [np.inf, -np.inf], 0).fillna(0) * 100
        
        kpi["Close Rate(MQL) %"] = (kpi["Close"] / kpi["Proposals"].replace(
        0, np.nan)).replace([np.inf, -np.inf], 0).fillna(0) * 100
    
    # ───────────────────────────────────────────────────────────────────────
    #  REVENUE METRICS
    # ───────────────────────────────────────────────────────────────────────
    # Always use the user-selected filter_column for consistency
    won_rev = fdate(op_won.copy(), filter_column)
    won_rev["Deal Value"] = pd.to_numeric(won_rev["Deal Value"],
                                  errors="coerce").fillna(0)
    
    rev_sum = won_rev.groupby("Owner")["Deal Value"].sum()
    kpi["Closed Revenue $"] = rev_sum.reindex(kpi.index, fill_value=0)
    
    with np.errstate(divide="ignore", invalid="ignore"):
        kpi["Revenue Per Call $"] = (kpi["Closed Revenue $"] /
                                 kpi["New Calls Booked"]).replace(
                                     [np.inf, -np.inf], 0).fillna(0)
        
        kpi["Revenue Per Showed Up $"] = (kpi["Closed Revenue $"] /
                                      kpi["Sales Call Taken"]).replace(
                                          [np.inf, -np.inf], 0).fillna(0)
        
        kpi["Revenue Per Proposal $"] = (
        kpi["Closed Revenue $"] /
        kpi["Proposals"].replace(0, np.nan)).replace([np.inf, -np.inf],
                                                     0).fillna(0)
    
    # Pipeline revenue (open proposals)
    pipe_rev = fdate(op_proposal.copy(), prop_date_col)
    pipe_rev["Deal Value"] = pd.to_numeric(pipe_rev["Deal Value"],
                                   errors="coerce").fillna(0)
    kpi["Pipeline Revenue $"] = (
    pipe_rev.groupby("Owner")["Deal Value"].sum().reindex(kpi.index,
                                                      fill_value=0))
    
    # ── TOTAL ROW  ────────────────────────────────────────────────────
    totals = {
    "Owner": "Total",
    "New Calls Booked": kpi["New Calls Booked"].sum(),
    "Sales Call Taken": kpi["Sales Call Taken"].sum(),
    "Proposals": kpi["Proposals"].sum(),
    "Show Rate %": kpi["Show Rate %"].mean(),
    "Unqualified": kpi["Unqualified"].sum(),
    "Unqualified Rate %": kpi["Unqualified Rate %"].mean(),
    "Cancelled Calls": kpi["Cancelled Calls"].sum(),
    "Cancellation Rate %": kpi["Cancellation Rate %"].mean(),
    "Proposal Rate %": kpi["Proposal Rate %"].mean(),
    "Close Rate %": kpi["Close Rate %"].mean(),
    "Close Rate(Show) %": kpi["Close Rate(Show) %"].mean(),
    "Close Rate(MQL) %": kpi["Close Rate(MQL) %"].mean(),
    "Closed Revenue $": kpi["Closed Revenue $"].sum(),
    "Revenue Per Call $": kpi["Revenue Per Call $"].mean(),
    "Revenue Per Showed Up $": kpi["Revenue Per Showed Up $"].mean(),
    "Revenue Per Proposal $": kpi["Revenue Per Proposal $"].mean(),
    "Pipeline Revenue $": kpi["Pipeline Revenue $"].sum(),
    "Cold Emails": kpi["Cold Emails"].sum(),
    #"LinkedIn": kpi["LinkedIn"].sum(),
    }
    
    # Create totals row with proper index
    totals_df = pd.DataFrame([totals])
    
    # Concatenate the main KPI data with totals
    kpi_final = pd.concat([kpi.reset_index(drop=True), totals_df],
                  ignore_index=True).drop(columns=["Close"],
                                          errors="ignore")
    return kpi_final



