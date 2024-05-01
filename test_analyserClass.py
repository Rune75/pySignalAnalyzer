def test_getTHDN():
    # Create an instance of the analyserClass
    analyser = analyserClass()

    # Set the necessary attributes for testing
    analyser.THDN_FS = 100
    analyser.getTHDN_FS = MagicMock(return_value=50)
    analyser.getFundamental().power = 10

    # Call the method under test
    result = analyser.getTHDN()

    # Check the result
    assert result == 40