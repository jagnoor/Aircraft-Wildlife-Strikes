import pytest
import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'Data')


@pytest.fixture(scope="module")
def database_df():
    """Load the main database once for all tests."""
    return pd.read_csv(os.path.join(DATA_DIR, 'database.csv'), low_memory=False)


@pytest.fixture(scope="module")
def subset_df():
    """Load the subset CSV once for all tests."""
    return pd.read_csv(os.path.join(DATA_DIR, 'data_subset.csv'), low_memory=False)


def test_database_loads_with_expected_shape(database_df):
    """Test 1: Verify database.csv has 174,104 rows and 66 columns."""
    assert database_df.shape[0] == 174104, f"Expected 174104 rows, got {database_df.shape[0]}"
    assert database_df.shape[1] == 66, f"Expected 66 columns, got {database_df.shape[1]}"


def test_subset_columns_match_expected(subset_df):
    """Test 2: Verify data_subset.csv has the correct columns."""
    expected_columns = [
        "Incident Year", "Incident Month", "Airport", "Airport ID",
        "State", "Aircraft", "Aircraft Type", "Flight Phase",
        "Species Name", "Species ID", "Species Quantity"
    ]
    for col in expected_columns:
        assert col in subset_df.columns, f"Missing expected column: {col}"


def test_flight_phase_groupby_and_concat(database_df):
    """Test 3: Reproduce cell 13 logic with pd.concat fix, verify OTHER category."""
    subset_df = database_df[["Incident Year", "Incident Month", "Airport", "Airport ID",
                              "State", "Aircraft", "Aircraft Type", "Flight Phase",
                              "Species Name", "Species ID", "Species Quantity"]]

    flightphase = subset_df.groupby(['Flight Phase'])
    flightphase = flightphase['Incident Year'].count().reset_index().rename(columns={"Incident Year": "Count"})
    flightphase = flightphase.sort_values(by='Count', ascending=False).reset_index(drop=True)

    total_before = flightphase['Count'].sum()

    others_total = flightphase.Count[6:].sum()
    new_row = {'Flight Phase': 'OTHER', 'Count': others_total}
    flightphase = pd.concat([flightphase, pd.DataFrame([new_row])], ignore_index=True)
    flightphase = flightphase.sort_values(by='Count', ascending=False).reset_index(drop=True)

    # Verify OTHER category exists
    assert 'OTHER' in flightphase['Flight Phase'].values, "OTHER category not created"
    # Verify we have 7 rows for the pie chart (top 6 + OTHER)
    assert len(flightphase) >= 7, f"Expected at least 7 rows, got {len(flightphase)}"
    # Verify Approach is the top flight phase
    assert flightphase.iloc[0]['Flight Phase'] == 'APPROACH', "Expected APPROACH as top flight phase"


def test_animal_kingdom_classification(database_df):
    """Test 4: Verify species ID classification logic for BIRD/MAMMAL/REPTILE."""
    subset_df = database_df[["Species ID", "Species Name"]].copy()

    subset_df["Animal_Kingdom"] = "BIRD"
    subset_df["Animal_Kingdom"] = np.where(
        subset_df['Species ID'].str.startswith('1', na=False),
        "MAMMAL", subset_df["Animal_Kingdom"]
    )
    subset_df["Animal_Kingdom"] = np.where(
        subset_df['Species ID'].str.startswith('2', na=False),
        "REPTILE", subset_df["Animal_Kingdom"]
    )

    valid_kingdoms = {'BIRD', 'MAMMAL', 'REPTILE'}
    actual_kingdoms = set(subset_df["Animal_Kingdom"].unique())
    assert actual_kingdoms == valid_kingdoms, f"Unexpected kingdoms: {actual_kingdoms}"

    # BIRD should be dominant (>90%)
    bird_pct = (subset_df["Animal_Kingdom"] == "BIRD").mean() * 100
    assert bird_pct > 90, f"Expected BIRD > 90%, got {bird_pct:.1f}%"

    # MAMMAL should be ~3%
    mammal_pct = (subset_df["Animal_Kingdom"] == "MAMMAL").mean() * 100
    assert 1 < mammal_pct < 10, f"Expected MAMMAL between 1-10%, got {mammal_pct:.1f}%"


def test_incident_year_range_and_no_nulls(database_df):
    """Test 5: Verify year range is 1990-2015 with no null values."""
    years = database_df["Incident Year"]

    assert years.notnull().all(), "Found null values in Incident Year"
    assert years.min() == 1990, f"Expected min year 1990, got {years.min()}"
    assert years.max() == 2015, f"Expected max year 2015, got {years.max()}"

    # Verify more recent years have more data (upward trend)
    early_count = len(database_df[database_df["Incident Year"] <= 2000])
    late_count = len(database_df[database_df["Incident Year"] > 2000])
    assert late_count > early_count, "Expected more incidents in later years"
