import unittest
import DecisionTreeModel as dtm


class TestDecisionTreeModel(unittest.TestCase):

    def test_predict(self):

        # Given a node and grow tree recursively until it meets the stop requirements.
        humidity = [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0]  # high = 0
        wind = [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1]
        play_tennis = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
        xTrains = [[h, w] for h, w in zip(humidity, wind)]
        model = dtm.DecisionTreeModel()
        model.fit(xTrains, play_tennis, 1)
        expected_predictions = [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0]
        # give me perfect prediction
        predictions = model.predict(xTrains)
        print(expected_predictions)
        print(predictions)
        self.assertTrue(predictions == expected_predictions)


if __name__ == '__main__':
    unittest.main()
