from mescal.study_manager import StudyManager
from mescal.mock.data_set import MockDataSet


mock_study_manager = StudyManager.factory_from_scenarios(
    scenarios=[MockDataSet('Base'), MockDataSet('Scen01'), MockDataSet('Scen02')],
    comparisons=[('Scen01', 'Base'), ('Scen02', 'Base')],
    export_folder='_tmp'
)


if __name__ == '__main__':
    print(mock_study_manager.scen.accepted_flags)
    print(mock_study_manager.scen.fetch('Node.Price'))
    print(mock_study_manager.comp.fetch('Node.Price'))
