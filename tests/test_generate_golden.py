from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import StylingConfig

styling_config = StylingConfig(
  input_format="Questions in English that asks for data in database.",
  expected_output_format="SQL query based on the given input",
  task="Answering text-to-SQL-related queries by querying a database and returning the results to users",
  scenario="Non-technical users trying to query a database using plain English.",
)


def test_generate_goldens_from_scratch():
    synthesizer = Synthesizer(styling_config=styling_config)
    goldens = synthesizer.generate_goldens_from_scratch(num_goldens=2)

    assert len(goldens) == 2
    for golden in goldens:
        assert golden.input
