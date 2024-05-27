from amltk.types import Space
import ConfigSpace
import copy


def make_subspaces_by_conditions(space: Space) -> tuple:
    space = copy.deepcopy(space)
    sub_spaces = []
    main_conditions = [
        item
        for item in space.get_conditions()
        if type(item) == ConfigSpace.conditions.EqualsCondition
    ]
    sub_conditions = [
        item
        for item in space.get_conditions()
        if type(item) != ConfigSpace.conditions.EqualsCondition
    ]
    conditions_parents = []
    conditions_values = []
    for item in main_conditions:
        if item.value not in conditions_values:
            conditions_values.append(item.value)
            conditions_parents.append(item.parent.name)
    conditions_values, conditions_parents = (
        list(t) for t in zip(*sorted(zip(conditions_values, conditions_parents)))
    )  # Sort them by alphabet
    for conditions_value, condition_name in zip(conditions_values, conditions_parents):
        dict_sub_space = {
            k: dict(space)[k] for k in dict(space) if conditions_value in k
        }
        sub_space = ConfigSpace.configuration_space.ConfigurationSpace(
            name=conditions_value, space=dict_sub_space
        )
        for item in sub_conditions:
            if conditions_value in str(item):
                for sub_item in item.components:
                    if condition_name not in str(sub_item):
                        sub_space.add_condition(sub_item)
        sub_spaces.append(sub_space)
    return sub_spaces, conditions_values, conditions_parents


def make_initial_config(space: Space) -> list:
    initial_configs = []
    for i, (sub_space, condition_name, parent_name) in enumerate(
        zip(*make_subspaces_by_conditions(space))
    ):
        config = dict(sub_space.get_default_configuration())
        config[parent_name] = condition_name
        initial_configs.append(
            ConfigSpace.configuration_space.Configuration(space, values=config)
        )
    return initial_configs
