"""Test of NPIV implementation against R npiv package results."""

import numpy as np

from moderndid.data import load_engel
from moderndid.didcont.npiv import npiv


def test_npiv_comprehensive_against_r():
    engel_data = load_engel()
    engel_data = engel_data.sort_values("logexp")

    food = engel_data["food"].values
    logexp = engel_data["logexp"].values
    logwages = engel_data["logwages"].values

    x_eval = np.linspace(4.5, 6.5, 100)

    result = npiv(
        y=food,
        x=logexp.reshape(-1, 1),
        w=logwages.reshape(-1, 1),
        x_eval=x_eval.reshape(-1, 1),
        j_x_degree=3,
        j_x_segments=1,
        k_w_degree=4,
        k_w_segments=4,
        knots="uniform",
        deriv_index=1,
        deriv_order=1,
        ucb_h=True,
        ucb_deriv=True,
        boot_num=99,
        seed=42,
    )

    expected_h = np.array(
        [
            0.2613976,
            0.2602563,
            0.2591179,
            0.2579822,
            0.2568490,
            0.2557181,
            0.2545892,
            0.2534623,
            0.2523369,
            0.2512130,
            0.2500903,
            0.2489686,
            0.2478477,
            0.2467273,
            0.2456073,
            0.2444874,
            0.2433674,
            0.2422471,
            0.2411263,
            0.2400048,
            0.2388823,
            0.2377587,
            0.2366336,
            0.2355070,
            0.2343786,
            0.2332482,
            0.2321155,
            0.2309803,
            0.2298425,
            0.2287018,
            0.2275580,
            0.2264109,
            0.2252603,
            0.2241059,
            0.2229475,
            0.2217849,
            0.2206180,
            0.2194464,
            0.2182701,
            0.2170886,
            0.2159019,
            0.2147098,
            0.2135119,
            0.2123082,
            0.2110983,
            0.2098821,
            0.2086593,
            0.2074298,
            0.2061933,
            0.2049496,
            0.2036985,
            0.2024397,
            0.2011731,
            0.1998985,
            0.1986155,
            0.1973241,
            0.1960240,
            0.1947149,
            0.1933967,
            0.1920692,
            0.1907320,
            0.1893851,
            0.1880282,
            0.1866611,
            0.1852836,
            0.1838953,
            0.1824963,
            0.1810861,
            0.1796647,
            0.1782318,
            0.1767871,
            0.1753305,
            0.1738617,
            0.1723806,
            0.1708868,
            0.1693803,
            0.1678607,
            0.1663279,
            0.1647817,
            0.1632218,
            0.1616480,
            0.1600601,
            0.1584579,
            0.1568412,
            0.1552098,
            0.1535633,
            0.1519017,
            0.1502248,
            0.1485322,
            0.1468237,
            0.1450993,
            0.1433586,
            0.1416014,
            0.1398276,
            0.1380368,
            0.1362289,
            0.1344037,
            0.1325610,
            0.1307005,
            0.1288220,
        ]
    )

    expected_deriv = np.array(
        [
            -0.05657280,
            -0.05642227,
            -0.05628240,
            -0.05615321,
            -0.05603469,
            -0.05592684,
            -0.05582965,
            -0.05574314,
            -0.05566730,
            -0.05560213,
            -0.05554764,
            -0.05550381,
            -0.05547065,
            -0.05544816,
            -0.05543635,
            -0.05543520,
            -0.05544473,
            -0.05546493,
            -0.05549579,
            -0.05553733,
            -0.05558954,
            -0.05565242,
            -0.05572597,
            -0.05581019,
            -0.05590508,
            -0.05601064,
            -0.05612687,
            -0.05625378,
            -0.05639135,
            -0.05653960,
            -0.05669851,
            -0.05686810,
            -0.05704835,
            -0.05723928,
            -0.05744088,
            -0.05765315,
            -0.05787609,
            -0.05810970,
            -0.05835398,
            -0.05860893,
            -0.05887455,
            -0.05915085,
            -0.05943781,
            -0.05973544,
            -0.06004375,
            -0.06036272,
            -0.06069237,
            -0.06103269,
            -0.06138368,
            -0.06174534,
            -0.06211766,
            -0.06250066,
            -0.06289434,
            -0.06329868,
            -0.06371369,
            -0.06413937,
            -0.06457573,
            -0.06502275,
            -0.06548044,
            -0.06594881,
            -0.06642785,
            -0.06691755,
            -0.06741793,
            -0.06792898,
            -0.06845070,
            -0.06898309,
            -0.06952615,
            -0.07007988,
            -0.07064428,
            -0.07121936,
            -0.07180510,
            -0.07240151,
            -0.07300860,
            -0.07362635,
            -0.07425478,
            -0.07489388,
            -0.07554364,
            -0.07620408,
            -0.07687519,
            -0.07755697,
            -0.07824942,
            -0.07895254,
            -0.07966633,
            -0.08039080,
            -0.08112593,
            -0.08187173,
            -0.08262821,
            -0.08339535,
            -0.08417317,
            -0.08496165,
            -0.08576081,
            -0.08657064,
            -0.08739114,
            -0.08822231,
            -0.08906415,
            -0.08991666,
            -0.09077984,
            -0.09165369,
            -0.09253822,
            -0.09343341,
        ]
    )

    np.testing.assert_allclose(
        result.h, expected_h, rtol=1e-3, atol=1e-5, err_msg="Function estimates (h) don't match R implementation"
    )

    np.testing.assert_allclose(
        result.deriv, expected_deriv, rtol=1e-3, atol=1e-5, err_msg="Derivative estimates don't match R implementation"
    )

    assert np.all(result.deriv < 0), "All derivatives should be negative"

    diff_deriv = np.diff(result.deriv)
    assert np.sum(diff_deriv <= 0) > 80, "Derivative should be mostly decreasing (>80%)"


def test_npiv_confidence_bands_against_r():
    engel_data = load_engel()
    engel_data = engel_data.sort_values("logexp")

    food = engel_data["food"].values
    logexp = engel_data["logexp"].values
    logwages = engel_data["logwages"].values

    x_eval = np.linspace(4.5, 6.5, 100)

    result = npiv(
        y=food,
        x=logexp.reshape(-1, 1),
        w=logwages.reshape(-1, 1),
        x_eval=x_eval.reshape(-1, 1),
        j_x_degree=3,
        j_x_segments=1,
        k_w_degree=4,
        k_w_segments=4,
        knots="uniform",
        ucb_h=True,
        ucb_deriv=True,
        boot_num=99,
        seed=42,
    )

    expected_h_lower = np.array(
        [
            0.18051009,
            0.18398483,
            0.18721942,
            0.19021683,
            0.19298016,
            0.19551267,
            0.19781784,
            0.19989945,
            0.20176173,
            0.20340939,
            0.20484779,
            0.20608303,
            0.20712207,
            0.20797281,
            0.20864414,
            0.20914596,
            0.20948915,
            0.20968544,
            0.20974726,
            0.20968755,
            0.20951944,
            0.20925607,
            0.20891025,
            0.20849420,
            0.20801938,
            0.20749626,
            0.20693422,
            0.20634142,
            0.20572478,
            0.20508993,
            0.20444125,
            0.20378189,
            0.20311377,
            0.20243768,
            0.20175329,
            0.20105920,
            0.20035302,
            0.19963137,
            0.19889003,
            0.19812391,
            0.19732725,
            0.19649368,
            0.19561643,
            0.19468850,
            0.19370297,
            0.19265327,
            0.19153349,
            0.19033873,
            0.18906538,
            0.18771130,
            0.18627598,
            0.18476051,
            0.18316751,
            0.18150091,
            0.17976570,
            0.17796767,
            0.17611314,
            0.17420868,
            0.17226098,
            0.17027660,
            0.16826191,
            0.16622300,
            0.16416558,
            0.16209500,
            0.16001617,
            0.15793360,
            0.15585134,
            0.15377299,
            0.15170172,
            0.14964018,
            0.14759056,
            0.14555450,
            0.14353309,
            0.14152683,
            0.13953554,
            0.13755835,
            0.13559361,
            0.13363878,
            0.13169041,
            0.12974400,
            0.12779392,
            0.12583330,
            0.12385398,
            0.12184641,
            0.11979960,
            0.11770110,
            0.11553705,
            0.11329228,
            0.11095044,
            0.10849425,
            0.10590582,
            0.10316700,
            0.10025981,
            0.09716683,
            0.09387159,
            0.09035892,
            0.08661517,
            0.08262839,
            0.07838829,
            0.07388626,
        ]
    )

    expected_h_upper = np.array(
        [
            0.3422852,
            0.3365277,
            0.3310163,
            0.3257475,
            0.3207178,
            0.3159235,
            0.3113607,
            0.3070251,
            0.3029121,
            0.2990166,
            0.2953328,
            0.2918541,
            0.2885732,
            0.2854817,
            0.2825704,
            0.2798287,
            0.2772456,
            0.2748087,
            0.2725053,
            0.2703220,
            0.2682451,
            0.2662612,
            0.2643570,
            0.2625199,
            0.2607378,
            0.2590001,
            0.2572967,
            0.2556193,
            0.2539603,
            0.2523137,
            0.2506748,
            0.2490399,
            0.2474067,
            0.2457740,
            0.2441417,
            0.2425107,
            0.2408830,
            0.2392615,
            0.2376501,
            0.2360533,
            0.2344766,
            0.2329259,
            0.2314074,
            0.2299279,
            0.2284937,
            0.2271109,
            0.2257852,
            0.2245209,
            0.2233212,
            0.2221879,
            0.2211210,
            0.2201189,
            0.2191787,
            0.2182960,
            0.2174654,
            0.2166805,
            0.2159348,
            0.2152211,
            0.2145324,
            0.2138617,
            0.2132022,
            0.2125473,
            0.2118909,
            0.2112272,
            0.2105509,
            0.2098571,
            0.2091412,
            0.2083993,
            0.2076277,
            0.2068233,
            0.2059836,
            0.2051064,
            0.2041903,
            0.2032343,
            0.2022381,
            0.2012022,
            0.2001279,
            0.1990171,
            0.1978730,
            0.1966996,
            0.1955021,
            0.1942870,
            0.1930619,
            0.1918360,
            0.1906199,
            0.1894256,
            0.1882664,
            0.1871572,
            0.1861139,
            0.1851532,
            0.1842928,
            0.1835502,
            0.1829430,
            0.1824883,
            0.1822020,
            0.1820990,
            0.1821923,
            0.1824936,
            0.1830127,
            0.1837578,
        ]
    )

    assert result.h_lower is not None, "Lower confidence band for h should exist"
    assert result.h_upper is not None, "Upper confidence band for h should exist"

    np.testing.assert_allclose(
        result.h_lower,
        expected_h_lower,
        rtol=0.3,
        atol=0.03,
        err_msg="h_lower confidence bands don't match R implementation within reasonable tolerance",
    )

    np.testing.assert_allclose(
        result.h_upper,
        expected_h_upper,
        rtol=0.3,
        atol=0.03,
        err_msg="h_upper confidence bands don't match R implementation within reasonable tolerance",
    )

    assert np.all(result.h_lower <= result.h), "Lower band should be <= estimate"
    assert np.all(result.h <= result.h_upper), "Estimate should be <= upper band"

    band_width = result.h_upper - result.h_lower
    assert np.all(band_width > 0.01), "Confidence bands should have meaningful width"
    assert np.all(band_width < 0.3), "Confidence bands shouldn't be too wide"


def test_npiv_derivative_confidence_bands_structure():
    engel_data = load_engel()
    engel_data = engel_data.sort_values("logexp")

    food = engel_data["food"].values
    logexp = engel_data["logexp"].values
    logwages = engel_data["logwages"].values

    x_eval = np.linspace(4.5, 6.5, 100)

    result = npiv(
        y=food,
        x=logexp.reshape(-1, 1),
        w=logwages.reshape(-1, 1),
        x_eval=x_eval.reshape(-1, 1),
        j_x_degree=3,
        j_x_segments=1,
        k_w_degree=4,
        k_w_segments=4,
        knots="uniform",
        ucb_h=True,
        ucb_deriv=True,
        boot_num=99,
        seed=42,
    )

    assert result.h_lower_deriv is not None, "Lower deriv confidence band should exist"
    assert result.h_upper_deriv is not None, "Upper deriv confidence band should exist"

    assert result.h_lower_deriv.shape == (100,), "h_lower_deriv should have 100 points"
    assert result.h_upper_deriv.shape == (100,), "h_upper_deriv should have 100 points"

    assert np.all(result.h_lower_deriv <= result.deriv), "Lower deriv <= estimate"
    assert np.all(result.deriv <= result.h_upper_deriv), "Estimate <= upper deriv"

    assert np.all(result.h_lower_deriv < 0), "Lower deriv band should be negative"
    assert np.mean(result.h_lower_deriv) < np.mean(result.deriv), "Lower deriv should be more negative on average"

    assert np.any(result.h_upper_deriv > 0), "Upper deriv should have some positive values"
    assert np.any(result.h_upper_deriv < 0), "Upper deriv should have some negative values"

    assert np.all(np.abs(result.h_lower_deriv) < 0.5), "Lower deriv bands should be reasonable"
    assert np.all(np.abs(result.h_upper_deriv) < 0.5), "Upper deriv bands should be reasonable"
