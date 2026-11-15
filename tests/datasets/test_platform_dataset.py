import dataclasses

import pandas as pd
import pytest

from mesqual.datasets import AmbiguousOverrideError, DatasetConcatCollection
from mesqual.flag import Dimension, UnknownFlagError
from mesqual.units import Units
from tests.datasets.fixtures import Interpreter, Platform, hourly_frame, series_spec

PRICE = 'Node.Results.price'
VOLUME = 'Node.Results.volume'
MEAN_PRICE = 'Node.Results.mean_price'


def prices() -> pd.DataFrame:
    return hourly_frame(periods=4) * 1000


class NodePriceInterpreter(Interpreter):
    FLAGS = frozenset({PRICE, VOLUME})

    def _fetch(self, flag, effective_config, **kwargs):
        if flag == PRICE:
            return self.prices
        return self.prices * 0 + 1


class MeanPriceInterpreter(Interpreter):
    FLAGS = frozenset({MEAN_PRICE})

    def _fetch(self, flag, effective_config, **kwargs):
        return self.parent_dataset.fetch(PRICE, effective_config).mean(axis=1).to_frame('mean')


class PriceCapInterpreter(NodePriceInterpreter):
    def _fetch(self, flag, effective_config, **kwargs):
        result = super()._fetch(flag, effective_config, **kwargs)
        return result.clip(upper=1000) if flag == PRICE else result


class PriceFloorInterpreter(PriceCapInterpreter):
    def _fetch(self, flag, effective_config, **kwargs):
        result = super()._fetch(flag, effective_config, **kwargs)
        return result.clip(lower=10) if flag == PRICE else result


def platform_classes():
    """A fresh platform class and study subclass with the base interpreters registered."""

    class MyPlatformDataset(Platform):
        pass

    class MyStudyDataset(MyPlatformDataset):
        pass

    MyPlatformDataset.register_interpreter(NodePriceInterpreter)
    MyPlatformDataset.register_interpreter(MeanPriceInterpreter)
    return MyPlatformDataset, MyStudyDataset


def study_with_price_cap():
    platform, study = platform_classes()
    study.register_interpreter(PriceCapInterpreter, override_interpreter=NodePriceInterpreter, version=1)
    return platform, study


def deeper_study_with_price_floor(version: int = 1):
    _, study = study_with_price_cap()

    class DeeperStudy(study):
        pass

    DeeperStudy.register_interpreter(PriceFloorInterpreter, override_interpreter=PriceCapInterpreter, version=version)
    return DeeperStudy


def sibling_studies():
    """A platform with the original, and two sibling subclasses that override it with two different caps."""
    platform, _ = platform_classes()
    left = type('Left', (platform,), {})
    right = type('Right', (platform,), {})
    other_cap = type('OtherCap', (PriceCapInterpreter,), {})
    left.register_interpreter(PriceCapInterpreter, override_interpreter=NodePriceInterpreter, version=1)
    right.register_interpreter(other_cap, override_interpreter=NodePriceInterpreter, version=1)
    return left, right, other_cap


class TestRegistry:
    def test_registration_on_subclass_does_not_leak(self):
        class Base(Platform):
            pass

        class Left(Base):
            pass

        class Right(Base):
            pass

        Left.register_interpreter(NodePriceInterpreter)
        assert Base.get_registered_interpreters() == []
        assert Right.get_registered_interpreters() == []
        assert Platform.get_registered_interpreters() == []
        assert Left.get_registered_interpreters() == [NodePriceInterpreter]

    def test_late_parent_registration_reaches_subclass(self):
        class Base(Platform):
            pass

        class Child(Base):
            pass

        Child.register_interpreter(MeanPriceInterpreter)
        Base.register_interpreter(NodePriceInterpreter)
        assert set(Child.get_registered_interpreters()) == {NodePriceInterpreter, MeanPriceInterpreter}
        assert Child(prices=prices()).fetch(MEAN_PRICE) is not None

    def test_instance_resolves_registry_once(self):
        class Base(Platform):
            pass

        Base.register_interpreter(NodePriceInterpreter)
        ds = Base(prices=prices())
        Base.register_interpreter(MeanPriceInterpreter)
        assert MEAN_PRICE not in ds.accepted_flags
        assert MEAN_PRICE in Base(prices=prices()).accepted_flags

    def test_decorator_forms(self):
        class Base(Platform):
            pass

        @Base.register_interpreter
        class A(NodePriceInterpreter):
            pass

        @Base.register_interpreter(override_interpreter=A, version=1)
        class B(A):
            pass

        assert Base.get_registered_interpreters() == [B]

    def test_wrong_interpreter_type_raises(self):
        class NotAnInterpreter(pd.DataFrame):
            pass

        with pytest.raises(TypeError):
            Platform.register_interpreter(NotAnInterpreter)
        with pytest.raises(TypeError):
            Platform.register_interpreter(NodePriceInterpreter, override_interpreter=NotAnInterpreter, version=1)


class TestFetchHook:
    def test_fetch_processes_the_result_of_the_answering_interpreter(self):
        platform, study = study_with_price_cap()

        class RoundedStudy(study):
            def _fetch(self, flag, effective_config, **kwargs):
                result = super()._fetch(flag, effective_config, **kwargs)
                return result.round(-3) if flag == PRICE else result

        dataset = RoundedStudy(prices=prices() + 1)
        assert dataset.fetch(PRICE).max().max() == 1000
        assert dataset.fetch(PRICE).min().min() == 0
        assert dataset.fetch(MEAN_PRICE)['mean'].tolist() == dataset.fetch(PRICE).mean(axis=1).tolist()


class TestOverride:
    def test_override_answers_on_study_only(self):
        platform, study = study_with_price_cap()
        assert study(prices=prices()).fetch(PRICE).max().max() == 1000
        assert platform(prices=prices()).fetch(PRICE).max().max() == 3000

    def test_derived_flag_gets_override_result(self):
        _, study = study_with_price_cap()
        assert study(prices=prices()).fetch(MEAN_PRICE)['mean'].max() == 1000

    def test_replaced_interpreter_is_not_created(self):
        _, study = study_with_price_cap()
        ds = study(prices=prices())
        assert study.get_registered_interpreters() == [PriceCapInterpreter, MeanPriceInterpreter]
        assert ds.get_flags_by_interpreter() == {
            PriceCapInterpreter: {PRICE, VOLUME},
            MeanPriceInterpreter: {MEAN_PRICE},
        }
        assert (ds.fetch(VOLUME) == 1).all().all()
        with pytest.raises(ValueError, match='No interpreter instance'):
            ds.get_interpreter_instance(NodePriceInterpreter)

    def test_override_need_not_subclass_the_original(self):
        _, study = platform_classes()

        class PriceOnly(Interpreter):
            FLAGS = frozenset({PRICE})

            def _fetch(self, flag, effective_config, **kwargs):
                return self.prices * 0 + 7

        study.register_interpreter(PriceOnly, override_interpreter=NodePriceInterpreter, version=1)
        ds = study(prices=prices())
        assert (ds.fetch(PRICE) == 7).all().all()
        assert not ds.flag_is_accepted(VOLUME)
        assert not ds.flag_index.has_spec(VOLUME)

    def test_override_on_a_subclass_beats_the_parent_override(self):
        ds = deeper_study_with_price_floor()(prices=prices())
        result = ds.fetch(PRICE)
        assert result.max().max() == 1000
        assert result.min().min() == 10
        assert ds.flag_info(PRICE).chain == ('NodePriceInterpreter', 'PriceCapInterpreter', 'PriceFloorInterpreter')
        deeper = type(ds)
        assert deeper.get_answering_interpreter(PRICE) is PriceFloorInterpreter
        assert deeper.__mro__[1].get_answering_interpreter(PRICE) is PriceCapInterpreter

    def test_override_beats_the_original_on_one_class(self):
        class Both(Platform):
            pass

        Both.register_interpreter(PriceCapInterpreter, override_interpreter=NodePriceInterpreter, version=1)
        Both.register_interpreter(NodePriceInterpreter)
        assert Both(prices=prices()).fetch(PRICE).max().max() == 1000
        assert Both.get_answering_interpreter(PRICE) is PriceCapInterpreter

    def test_original_registered_again_below_an_override_keeps_the_override(self):
        _, study = study_with_price_cap()

        class Again(study):
            pass

        Again.register_interpreter(NodePriceInterpreter)
        ds = Again(prices=prices())
        assert Again.get_answering_interpreter(PRICE) is PriceCapInterpreter
        assert ds.fetch(PRICE).max().max() == 1000
        assert ds.flag_info(PRICE).chain == ('NodePriceInterpreter', 'PriceCapInterpreter')

    def test_plain_registration_of_an_override_below_keeps_the_chain(self):
        _, study = study_with_price_cap()
        lower = type('Lower', (study,), {})
        lower.register_interpreter(PriceCapInterpreter)
        assert lower.get_registered_interpreters() == [PriceCapInterpreter, MeanPriceInterpreter]
        assert lower(prices=prices()).flag_info(PRICE).chain == ('NodePriceInterpreter', 'PriceCapInterpreter')


class TestMultipleInheritance:
    def test_overrides_on_sibling_classes_raise_on_a_common_subclass(self):
        left, right, _ = sibling_studies()
        both = type('Both', (left, right), {})
        match = 'neither class is a subclass of the other'
        with pytest.raises(AmbiguousOverrideError, match=match) as error:
            both(prices=prices())
        assert 'PriceCapInterpreter (on Left) and OtherCap (on Right) both replace NodePriceInterpreter' in str(
            error.value
        )
        with pytest.raises(AmbiguousOverrideError, match=match):
            both.get_answering_interpreter(PRICE)
        assert left.get_answering_interpreter(PRICE) is PriceCapInterpreter

    @pytest.mark.parametrize('on_class_between', [False, True])
    def test_override_on_the_common_subclass_or_a_class_between_answers(self, on_class_between):
        left, right, other_cap = sibling_studies()
        both = type('Both', (left, right), {})
        bottom = type('Bottom', (both,), {}) if on_class_between else both
        both.register_interpreter(PriceFloorInterpreter, override_interpreter=other_cap, version=1)
        ds = bottom(prices=prices())
        assert bottom.get_answering_interpreter(PRICE) is PriceFloorInterpreter
        assert ds.flag_info(PRICE).chain == ('NodePriceInterpreter', 'OtherCap', 'PriceFloorInterpreter')
        assert ds.fetch(PRICE).min().min() == 10

    @pytest.mark.parametrize('bases', ['override first', 'original first'])
    def test_diamond_with_one_override_is_independent_of_the_base_order(self, bases):
        platform, _ = platform_classes()
        capped = type('Capped', (platform,), {})
        plain = type('Plain', (platform,), {})
        plain.register_interpreter(NodePriceInterpreter)
        capped.register_interpreter(PriceCapInterpreter, override_interpreter=NodePriceInterpreter, version=1)
        diamond = type('Diamond', (capped, plain) if bases == 'override first' else (plain, capped), {})
        assert diamond.get_answering_interpreter(PRICE) is PriceCapInterpreter
        assert diamond(prices=prices()).flag_info(PRICE).chain == ('NodePriceInterpreter', 'PriceCapInterpreter')

    @pytest.mark.parametrize('bases', ['version 1 first', 'version 2 first'])
    def test_override_of_an_interpreter_with_two_versions_on_sibling_classes_raises(self, bases):
        platform, _ = platform_classes()
        left = type('Left', (platform,), {})
        right = type('Right', (platform,), {})
        left.register_interpreter(PriceCapInterpreter, override_interpreter=NodePriceInterpreter, version=1)
        right.register_interpreter(PriceCapInterpreter, override_interpreter=NodePriceInterpreter, version=2)
        both = type('Both', (left, right) if bases == 'version 1 first' else (right, left), {})
        both.register_interpreter(PriceFloorInterpreter, override_interpreter=PriceCapInterpreter, version=1)
        with pytest.raises(
            AmbiguousOverrideError,
            match='register differently. Register PriceCapInterpreter once on a common parent class',
        ):
            both(prices=prices())

    @pytest.mark.parametrize('bases', ['version 1 first', 'version 2 first'])
    def test_one_override_with_two_versions_on_sibling_classes_raises_on_a_common_subclass(self, bases):
        platform, _ = platform_classes()
        left = type('Left', (platform,), {})
        right = type('Right', (platform,), {})
        left.register_interpreter(PriceCapInterpreter, override_interpreter=NodePriceInterpreter, version=1)
        right.register_interpreter(PriceCapInterpreter, override_interpreter=NodePriceInterpreter, version=2)
        both = type('Both', (left, right) if bases == 'version 1 first' else (right, left), {})
        with pytest.raises(AmbiguousOverrideError) as error:
            both(prices=prices())
        message = str(error.value)
        assert 'PriceCapInterpreter (on Left, version 1)' in message
        assert 'PriceCapInterpreter (on Right, version 2)' in message
        assert 'both replace NodePriceInterpreter' in message
        assert 'Register PriceCapInterpreter with one version on a common parent class' in message

    def test_sibling_conflict_names_the_interpreter_that_both_replace(self):
        _, study = study_with_price_cap()
        left = type('Left', (study,), {})
        right = type('Right', (study,), {})
        left.register_interpreter(PriceFloorInterpreter, override_interpreter=PriceCapInterpreter, version=1)
        right.register_interpreter(
            type('OtherFloor', (PriceCapInterpreter,), {}),
            override_interpreter=PriceCapInterpreter,
            version=1,
        )
        with pytest.raises(AmbiguousOverrideError, match='OtherFloor \\(on Right\\) both replace PriceCapInterpreter,'):
            type('Both', (left, right), {})(prices=prices())

    def test_override_of_an_original_that_siblings_replace_names_both_replacements(self):
        left, right, _ = sibling_studies()
        both = type('Both', (left, right), {})
        both.register_interpreter(PriceFloorInterpreter, override_interpreter=NodePriceInterpreter, version=1)
        with pytest.raises(ValueError, match='which PriceCapInterpreter or OtherCap replaces on a parent class'):
            both(prices=prices())

    def test_one_interpreter_that_replaces_two_originals_on_sibling_classes_names_them(self):
        platform, _ = platform_classes()
        left = type('Left', (platform,), {})
        right = type('Right', (platform,), {})
        left.register_interpreter(PriceCapInterpreter, override_interpreter=NodePriceInterpreter, version=1)
        right.register_interpreter(PriceCapInterpreter, override_interpreter=MeanPriceInterpreter, version=1)
        both = type('Both', (left, right), {})
        both.register_interpreter(PriceFloorInterpreter, override_interpreter=PriceCapInterpreter, version=1)
        with pytest.raises(
            AmbiguousOverrideError,
            match=r'Left \(replaces NodePriceInterpreter, version 1\) and Right \(replaces MeanPriceInterpreter, version 1\)',
        ):
            both(prices=prices())

    def test_class_view_and_instance_agree_in_any_registration_order(self):
        own_spec = dataclasses.replace(series_spec(PRICE), description='Capped price.')

        class Described(PriceCapInterpreter):
            flag_specs = (own_spec, series_spec(VOLUME))

        results = []
        for override_first in (True, False):

            class Ordered(Platform):
                pass

            registrations = [
                (Described, {'override_interpreter': NodePriceInterpreter, 'version': 1}),
                (NodePriceInterpreter, {}),
            ]
            for interpreter, kwargs in registrations if override_first else reversed(registrations):
                Ordered.register_interpreter(interpreter, **kwargs)
            class_spec = Ordered._class_flag_index().get_spec(PRICE)
            instance = Ordered(prices=prices())
            assert class_spec is instance.flag_index.get_spec(PRICE) is own_spec
            assert Ordered.describe_class_flags().loc[PRICE, 'description'] == 'Capped price.'
            results.append((Ordered.get_answering_interpreter(PRICE), instance.flag_info(PRICE).chain[-1]))
        assert results[0] == results[1] == (Described, Described.__qualname__)


class TestErrors:
    def test_override_of_an_unregistered_interpreter_raises(self):
        class Lonely(Platform):
            pass

        Lonely.register_interpreter(PriceCapInterpreter, override_interpreter=NodePriceInterpreter, version=1)
        with pytest.raises(ValueError, match='NodePriceInterpreter, which is not registered on Lonely or its parents'):
            Lonely(prices=prices())

    def test_override_on_a_parent_of_the_class_of_the_original_raises(self):
        class Parent(Platform):
            pass

        class Child(Parent):
            pass

        Parent.register_interpreter(PriceCapInterpreter, override_interpreter=NodePriceInterpreter, version=1)
        Child.register_interpreter(NodePriceInterpreter)
        with pytest.raises(ValueError, match='which is not registered on Parent or its parents'):
            Child(prices=prices())
        with pytest.raises(ValueError, match='which is not registered on Parent or its parents'):
            Child.get_answering_interpreter(PRICE)

    def test_override_of_a_replaced_original_names_the_interpreter_to_override_instead(self):
        _, study = study_with_price_cap()

        class Deeper(study):
            pass

        Deeper.register_interpreter(PriceFloorInterpreter, override_interpreter=NodePriceInterpreter, version=1)
        with pytest.raises(
            ValueError,
            match='which PriceCapInterpreter replaces on a parent class. Override PriceCapInterpreter instead',
        ):
            Deeper(prices=prices())

    @pytest.mark.parametrize('cap_first', [True, False])
    def test_override_of_an_override_on_the_same_class_raises_at_registration(self, cap_first):
        _, study = platform_classes()
        cap = (PriceCapInterpreter, NodePriceInterpreter)
        floor = (PriceFloorInterpreter, PriceCapInterpreter)
        first, second = (cap, floor) if cap_first else (floor, cap)
        study.register_interpreter(first[0], override_interpreter=first[1], version=1)
        message = (
            'PriceFloorInterpreter on MyStudyDataset overrides PriceCapInterpreter, '
            'which is an override on MyStudyDataset too. Register one of them on a subclass.'
        )
        with pytest.raises(ValueError, match=message):
            study.register_interpreter(second[0], override_interpreter=second[1], version=1)
        assert [r.interpreter for r in study._own_registrations()] == [first[0]]
        if cap_first:
            assert study.get_answering_interpreter(PRICE) is PriceCapInterpreter
        else:
            with pytest.raises(ValueError, match='PriceCapInterpreter, which is not registered on MyStudyDataset'):
                study.get_answering_interpreter(PRICE)

    @pytest.mark.parametrize(
        'arguments, message',
        [
            ({'override_interpreter': NodePriceInterpreter}, 'version is required with override_interpreter'),
            ({'version': 1}, 'version needs override_interpreter'),
            ({'override_interpreter': PriceCapInterpreter, 'version': 1}, 'names itself as override_interpreter'),
        ],
    )
    def test_invalid_override_arguments_raise(self, arguments, message):
        _, study = platform_classes()
        with pytest.raises(ValueError, match=message):
            study.register_interpreter(PriceCapInterpreter, **arguments)

    def test_two_overrides_of_one_interpreter_on_one_class_raise(self):
        _, study = platform_classes()

        class OtherCap(PriceCapInterpreter):
            pass

        study.register_interpreter(PriceCapInterpreter, override_interpreter=NodePriceInterpreter, version=1)
        message = (
            'PriceCapInterpreter and OtherCap both replace NodePriceInterpreter on MyStudyDataset. '
            'Register one of them on a subclass to override the other.'
        )
        with pytest.raises(AmbiguousOverrideError, match=message):
            study.register_interpreter(OtherCap, override_interpreter=NodePriceInterpreter, version=1)
        assert study.get_answering_interpreter(PRICE) is PriceCapInterpreter

    def test_two_overrides_on_one_class_of_two_conflicting_answers_raise(self):
        left, right, other_cap = sibling_studies()
        both = type('Both', (left, right), {})
        both.register_interpreter(PriceFloorInterpreter, override_interpreter=other_cap, version=1)
        both.register_interpreter(
            type('Floor', (PriceCapInterpreter,), {}), override_interpreter=PriceCapInterpreter, version=1
        )
        with pytest.raises(AmbiguousOverrideError, match='both replace NodePriceInterpreter on Both'):
            both(prices=prices())

    def test_override_that_declares_a_spec_of_another_interpreter_raises(self):
        _, study = platform_classes()

        class CapWithMean(PriceCapInterpreter):
            flag_specs = (series_spec(PRICE), series_spec(VOLUME), series_spec(MEAN_PRICE))

        study.register_interpreter(CapWithMean, override_interpreter=NodePriceInterpreter, version=1)
        with pytest.raises(AmbiguousOverrideError, match='both declare the spec'):
            study(prices=prices())

    def test_two_interpreters_with_the_same_spec_raise_at_creation_naming_both(self):
        class Twice(Platform):
            pass

        class First(Interpreter):
            FLAGS = frozenset({'x'})

        class Second(Interpreter):
            FLAGS = frozenset({'x'})

        Twice.register_interpreter(First)
        Twice.register_interpreter(Second)
        with pytest.raises(AmbiguousOverrideError, match='First and Second both declare'):
            Twice()
        with pytest.raises(AmbiguousOverrideError, match='First and Second both declare'):
            Twice.get_answering_interpreter('x')

    def test_two_interpreters_with_the_same_family_spec_raise_at_creation(self):
        class Dynamic(Platform):
            pass

        class First(Interpreter):
            flag_specs = (
                series_spec(
                    'd.{name}', variants=(Dimension('name', 'Name resolved per dataset.', lambda ds: {'x': 'x'}),)
                ),
            )

        class Second(First):
            pass

        Dynamic.register_interpreter(First)
        Dynamic.register_interpreter(Second)
        with pytest.raises(AmbiguousOverrideError, match='First and Second both declare'):
            Dynamic(prices=prices())

    def test_two_interpreters_with_the_same_qualname_raise_at_creation_naming_both_modules(self):
        class SameName(Platform):
            pass

        class First(Interpreter):
            FLAGS = frozenset({'x'})

        class Second(Interpreter):
            FLAGS = frozenset({'y'})

        for interpreter, module in ((First, 'plugins.left'), (Second, 'plugins.right')):
            interpreter.__qualname__ = 'Prices'
            interpreter.__module__ = module
        SameName.register_interpreter(First)
        SameName.register_interpreter(Second)
        with pytest.raises(
            ValueError, match=r"'Prices': one in module 'plugins\.left', one in module 'plugins\.right'"
        ):
            SameName()

    def test_same_qualname_from_one_module_raises(self):
        def make(flag):
            class Prices(Interpreter):
                FLAGS = frozenset({flag})

            return Prices

        class Factory(Platform):
            pass

        Factory.register_interpreter(make('x'))
        Factory.register_interpreter(make('y'))
        with pytest.raises(
            ValueError,
            match=r"make\.<locals>\.Prices': two different classes in module 'tests\.datasets\.test_platform_dataset'",
        ):
            Factory()

    def test_ambiguous_override_error_is_a_value_error(self):
        assert issubclass(AmbiguousOverrideError, ValueError)


class TestIntrospection:
    def test_get_flags_by_interpreter(self):
        platform, _ = platform_classes()
        flags = platform(prices=prices()).get_flags_by_interpreter()
        assert flags == {NodePriceInterpreter: {PRICE, VOLUME}, MeanPriceInterpreter: {MEAN_PRICE}}

    def test_get_interpreter_instance_unknown_type_raises(self):
        platform, _ = platform_classes()
        with pytest.raises(ValueError):
            platform(prices=prices()).get_interpreter_instance(PriceCapInterpreter)


class TestDynamicAcceptance:
    @staticmethod
    def prefix_platform():
        class PrefixInterpreter(Interpreter):
            flag_specs = (
                series_spec(
                    'a.{name}',
                    variants=(Dimension('name', 'Name resolved per dataset.', lambda ds: {'b': 'b', 'c': 'c'}),),
                ),
            )

            def _fetch(self, flag, effective_config, **kwargs):
                return pd.DataFrame({'flag': [flag]})

        class Prefixed(Platform):
            pass

        Prefixed.register_interpreter(PrefixInterpreter)
        return Prefixed, PrefixInterpreter

    def test_family_flag_routes_by_flag_is_accepted(self):
        platform, _ = self.prefix_platform()
        ds = platform(prices=prices())
        assert ds.flag_is_accepted('a.c')
        assert not ds.flag_is_accepted('b.c')
        assert ds.fetch('a.c')['flag'].iloc[0] == 'a.c'

    def test_family_flag_outside_the_dataset_values_raises_at_fetch(self):
        platform, _ = self.prefix_platform()
        ds = platform(prices=prices())
        assert ds.flag_is_accepted('a.z')
        with pytest.raises(ValueError, match="'z' is not a value of 'name'"):
            ds.fetch('a.z')

    def test_override_of_a_family_interpreter_answers_every_flag_of_the_family(self):
        platform, base = self.prefix_platform()

        @platform.register_interpreter(override_interpreter=base, version=1)
        class Upper(base):
            def _fetch(self, flag, effective_config, **kwargs):
                return super()._fetch(flag, effective_config, **kwargs).apply(lambda c: c.str.upper())

        ds = platform(prices=prices())
        assert platform.get_answering_interpreter('a.c') is Upper
        assert [ds.fetch(flag)['flag'].iloc[0] for flag in ('a.b', 'a.c')] == ['A.B', 'A.C']


class TestRegistrationDetails:
    def test_same_registration_twice_is_a_no_op(self):
        platform, _ = platform_classes()
        platform.register_interpreter(NodePriceInterpreter)
        assert platform.get_registered_interpreters() == [NodePriceInterpreter, MeanPriceInterpreter]

    def test_same_interpreter_with_other_arguments_raises(self):
        _, study = study_with_price_cap()
        with pytest.raises(ValueError, match='other arguments'):
            study.register_interpreter(PriceCapInterpreter, override_interpreter=NodePriceInterpreter, version=2)


class TestLazyAcceptedFlags:
    def test_accepted_flags_can_look_up_a_sibling(self):
        class Sibling(Platform):
            pass

        class FlagSource(Interpreter):
            FLAGS = frozenset({'x.one', 'x.two'})

            def _fetch(self, flag, effective_config, **kwargs):
                return pd.DataFrame({'flag': [flag]})

        class Mirror(Interpreter):
            flag_specs = (series_spec('mirror.one'), series_spec('mirror.two'))

            @property
            def accepted_flags(self) -> set:
                source = self.parent_dataset.get_dataset_by_type(FlagSource)
                return {flag.replace('x.', 'mirror.') for flag in source.accepted_flags}

            def _fetch(self, flag, effective_config, **kwargs):
                return self.parent_dataset.fetch(flag.replace('mirror.', 'x.'), effective_config)

        Sibling.register_interpreter(Mirror)
        Sibling.register_interpreter(FlagSource)
        ds = Sibling(prices=prices())
        assert ds.accepted_flags == {'x.one', 'x.two', 'mirror.one', 'mirror.two'}
        assert ds.fetch('mirror.two')['flag'].iloc[0] == 'x.two'

    def test_accepted_flags_can_read_attribute_set_after_super_init(self):
        class Late(Platform):
            pass

        class LateFlags(Interpreter):
            flag_specs = (series_spec('late.flag'),)

            def __init__(self, parent_dataset=None, prices=None):
                super().__init__(parent_dataset=parent_dataset, prices=prices)
                self.flags = {'late.flag'}

            @property
            def accepted_flags(self) -> set:
                return self.flags

            def _fetch(self, flag, effective_config, **kwargs):
                return pd.DataFrame({'flag': [flag]})

        Late.register_interpreter(LateFlags)
        assert Late(prices=prices()).fetch('late.flag')['flag'].iloc[0] == 'late.flag'

    def test_accepted_flags_are_returned_as_copies(self):
        class Copies(Platform):
            pass

        Copies.register_interpreter(NodePriceInterpreter)
        ds = Copies(prices=prices())
        ds.accepted_flags.clear()
        ds.get_flags_by_interpreter()[NodePriceInterpreter].clear()
        assert ds.accepted_flags == {PRICE, VOLUME}
        assert ds.get_flags_by_interpreter() == {NodePriceInterpreter: {PRICE, VOLUME}}


class TestReadmeOverridePattern:
    def test_enriching_existing_model_data(self):
        class ModelInterpreter(Interpreter):
            FLAGS = frozenset({'buses', 'lines'})

            def _fetch(self, flag, effective_config, **kwargs):
                return pd.DataFrame({'name': [f'{flag}_1', f'{flag}_2']})

        class ControlAreaModelInterpreter(Interpreter):
            FLAGS = frozenset({'control_areas'})

            def _fetch(self, flag, effective_config, **kwargs):
                return pd.DataFrame({'control_area': ['CA1', 'CA2']})

        class BusModelInterpreter(ModelInterpreter):
            def _fetch(self, flag, effective_config, **kwargs):
                df = super()._fetch(flag, effective_config, **kwargs)
                if flag == 'buses':
                    df['control_area'] = self.parent_dataset.fetch('control_areas', effective_config)['control_area']
                return df

        class PlatformData(Platform):
            pass

        class StudyData(PlatformData):
            pass

        PlatformData.register_interpreter(ModelInterpreter)
        StudyData.register_interpreter(ControlAreaModelInterpreter)
        StudyData.register_interpreter(BusModelInterpreter, override_interpreter=ModelInterpreter, version=1)

        ds = StudyData(prices=prices())
        assert list(ds.fetch('buses')['control_area']) == ['CA1', 'CA2']
        assert 'control_area' not in ds.fetch('lines')
        assert 'control_area' not in PlatformData(prices=prices()).fetch('buses')


SOLUTION = Dimension('solution', 'Solutions.', {'first': 'first', 'final': 'final'}, default='final')
CAP = Dimension('cap', 'Price cap in EUR/MWh.', default=1000)


class SolutionPrices(Interpreter):
    """Node prices with a `solution` parameter. It records the parameters that reach it."""

    flag_specs = (series_spec(PRICE, unit=Units.EUR_per_MWh, params=(SOLUTION,)),)
    seen: list = []

    def _fetch(self, flag, effective_config, solution='final', **params):
        SolutionPrices.seen.append({'solution': solution, **params})
        return self.prices + (0 if solution == 'final' else -1)


class CappedPrices(SolutionPrices):
    flag_specs = (series_spec(PRICE, unit=Units.EUR_per_MWh, params=(SOLUTION, CAP)),)

    def _fetch(self, flag, effective_config, cap=1000, **params):
        return super()._fetch(flag, effective_config, **params).clip(upper=cap)


def capped_prices_platform():
    class PricePlatform(Platform):
        pass

    class CapStudy(PricePlatform):
        pass

    PricePlatform.register_interpreter(SolutionPrices)
    CapStudy.register_interpreter(CappedPrices, override_interpreter=SolutionPrices, version=1)
    return PricePlatform, CapStudy


class TestOverrideSpecs:
    def test_added_parameter_with_default_reaches_the_override_only(self):
        platform, study = capped_prices_platform()
        SolutionPrices.seen.clear()
        ds = study(prices=prices())
        assert ds.fetch(PRICE).max().max() == 1000
        assert ds.fetch(PRICE, cap=2000).max().max() == 2000
        assert ds.fetch(PRICE, cap=2000, solution='first').max().max() == 2000
        assert SolutionPrices.seen == [{'solution': 'final'}, {'solution': 'final'}, {'solution': 'first'}]
        with pytest.raises(ValueError, match='cap'):
            platform(prices=prices()).fetch(PRICE, cap=2000)

    def test_added_parameter_is_part_of_the_record(self):
        _, study = capped_prices_platform()
        ds = study(prices=prices())
        ds.fetch(PRICE, cap=2000)
        assert ds.get_fetch_record(PRICE, cap=2000).params == {'solution': 'final', 'cap': 2000}

    def test_changed_unit_and_variants_give_the_flags_of_the_new_spec(self):
        side = Dimension('side', 'Side.', {'sent': 'sent', 'received': 'received'})
        flows = series_spec('flow_{side}', variants=(side,))
        flows_in_kw = series_spec(
            'flow_{side}',
            unit=Units.kW,
            variants=(dataclasses.replace(side, values={'sent': 'sent', 'net': 'net'}),),
        )

        class Flows(Interpreter):
            flag_specs = (flows,)

            def _fetch(self, flag, effective_config, **kwargs):
                return pd.DataFrame({'flag': [flag]})

        class NetFlowsInKW(Flows):
            flag_specs = (flows_in_kw,)

            def _fetch(self, flag, effective_config, **kwargs):
                return pd.DataFrame({'flag': [flag.upper()]})

        class FlowPlatform(Platform):
            pass

        class FlowStudy(FlowPlatform):
            pass

        FlowPlatform.register_interpreter(Flows)
        FlowStudy.register_interpreter(NetFlowsInKW, override_interpreter=Flows, version=1)
        ds = FlowStudy(prices=prices())
        assert ds.accepted_flags == {'flow_sent', 'flow_net'}
        assert ds.flag_index.get_unit('flow_net') == Units.kW
        assert ds.fetch('flow_net')['flag'].iloc[0] == 'FLOW_NET'
        assert not ds.flag_is_accepted('flow_received')
        assert FlowPlatform(prices=prices()).accepted_flags == {'flow_sent', 'flow_received'}

    def test_override_on_top_of_an_override_inherits_its_spec_by_subclassing(self):
        _, study = capped_prices_platform()

        class DeeperStudy(study):
            pass

        answering = study.get_answering_interpreter(PRICE)

        @DeeperStudy.register_interpreter(override_interpreter=answering, version=1)
        class FlooredPrices(answering):
            def _fetch(self, flag, effective_config, **params):
                return super()._fetch(flag, effective_config, **params).clip(lower=10)

        ds = DeeperStudy(prices=prices())
        assert ds.flag_index.get_spec(PRICE) is CappedPrices.flag_specs[0]
        result = ds.fetch(PRICE, cap=1500)
        assert (result.min().min(), result.max().max()) == (10, 1500)
        assert ds.flag_info(PRICE).chain == ('SolutionPrices', 'CappedPrices', FlooredPrices.__qualname__)

    def test_collection_of_a_base_and_a_compatible_override_dataset(self):
        platform, study = study_with_price_cap()
        collection = DatasetConcatCollection(
            [platform(name='base', prices=prices()), study(name='cap', prices=prices())]
        )
        result = collection.fetch(PRICE)
        assert result['base'].max().max() == 3000
        assert result['cap'].max().max() == 1000
        assert collection.fetch(MEAN_PRICE)['cap']['mean'].max() == 1000

    def test_collection_of_datasets_with_different_specs_of_one_name_raises(self):
        platform, study = capped_prices_platform()
        with pytest.raises(ValueError, match='Two different specs'):
            DatasetConcatCollection([platform(name='base', prices=prices()), study(name='cap', prices=prices())])


class TestAnsweringInterpreter:
    def test_lookup_without_overrides_gives_the_original(self):
        platform, study = platform_classes()
        assert platform.get_answering_interpreter(PRICE) is NodePriceInterpreter
        assert study.get_answering_interpreter(MEAN_PRICE) is MeanPriceInterpreter

    def test_lookup_of_a_flag_family(self):
        platform, base = TestDynamicAcceptance.prefix_platform()
        assert platform.get_answering_interpreter('a.z') is base

    def test_unknown_flag_raises_naming_the_class(self):
        platform, _ = platform_classes()
        with pytest.raises(UnknownFlagError, match='MyPlatformDataset'):
            platform.get_answering_interpreter('Node.Results.unknown')

    def test_platform_class_with_own_flag_specs_raises(self):
        with pytest.raises(TypeError, match='OwnSpecs sets flag_specs'):
            type('OwnSpecs', (Platform,), {'flag_specs': (series_spec('own'),)})


class TestVersionIdentity:
    def test_chain_gives_one_pair_per_link(self):
        ds = deeper_study_with_price_floor(version=3)(prices=prices())
        chain = (('NodePriceInterpreter', 1), ('PriceCapInterpreter', 1), ('PriceFloorInterpreter', 3))
        assert ds.version_identity(PRICE) == chain
        assert ds.version_identity(VOLUME) == chain
        assert ds.version_identity(MEAN_PRICE) == (('MeanPriceInterpreter', 1),)
        assert ds.get_interpreter_instance(PriceFloorInterpreter).version_identity(PRICE) == chain

    def test_first_link_takes_the_version_of_the_new_spec(self):
        _, study = platform_classes()

        class Versioned(PriceCapInterpreter):
            flag_specs = (series_spec(PRICE, version=4), series_spec(VOLUME))

        study.register_interpreter(Versioned, override_interpreter=NodePriceInterpreter, version=1)
        assert study(prices=prices()).version_identity(PRICE) == (
            ('NodePriceInterpreter', 4),
            (Versioned.__qualname__, 1),
        )


class TestNoSourceAccess:
    def test_accepted_flags_and_routing_open_no_source(self):
        calls = []

        class CountingInterpreter(Interpreter):
            flag_specs = (
                series_spec(PRICE),
                series_spec(
                    'gen_{fuel}',
                    variants=(Dimension('fuel', 'Fuel.', lambda ds: calls.append('values') or {'gas': 'gas'}),),
                ),
            )

            def _fetch(self, flag, effective_config, **kwargs):
                calls.append(flag)
                return prices()

        class Override(CountingInterpreter):
            pass

        class Counted(Platform):
            pass

        Counted.register_interpreter(CountingInterpreter)
        Counted.register_interpreter(Override, override_interpreter=CountingInterpreter, version=1)
        ds = Counted(prices=prices())
        assert ds.accepted_flags == {PRICE}
        assert ds.flag_is_accepted('gen_gas')
        assert ds.get_flags_by_interpreter() == {Override: {PRICE}}
        assert ds.flag_index.has_spec('gen_coal')
        assert calls == []


class TestConstructorArguments:
    @pytest.mark.parametrize('unknown', [['database'], ['cahce'], ['cahce', 'database']])
    def test_unknown_keyword_arguments_raise_naming_them_sorted_and_the_accepted_arguments(self, unknown):
        platform, _ = platform_classes()
        with pytest.raises(TypeError) as error:
            platform(prices=prices(), **{name: object() for name in reversed(unknown)})
        assert str(error.value) == (
            f'MyPlatformDataset got unknown keyword arguments {unknown}. '
            "Accepted by PlatformDataset.__init__: ['attributes', 'cache', 'config', 'name', 'prices'] "
            '(its own arguments and the constructor arguments of Interpreter).'
        )

    def test_unknown_keyword_argument_raises_before_the_interpreters_are_created(self):
        created = []

        class Recording(Interpreter):
            FLAGS = frozenset({PRICE})

            def __init__(self, parent_dataset=None, prices=None):
                created.append(self)
                super().__init__(parent_dataset=parent_dataset, prices=prices)

        class Checked(Platform):
            pass

        Checked.register_interpreter(Recording)
        with pytest.raises(TypeError):
            Checked(database=None)
        assert created == []

    def test_interpreter_argument_reaches_every_interpreter(self):
        platform, _ = platform_classes()
        given = prices()
        ds = platform(prices=given)
        assert ds.get_interpreter_instance(NodePriceInterpreter).prices is given
        assert ds.get_interpreter_instance(MeanPriceInterpreter).prices is given

    def test_subclass_takes_its_own_argument_before_super_init(self):
        class FolderPlatform(Platform):
            def __init__(self, folder: str, **kwargs):
                self.folder = folder
                super().__init__(**kwargs)

        FolderPlatform.register_interpreter(NodePriceInterpreter)
        assert FolderPlatform('results', prices=prices()).folder == 'results'
        with pytest.raises(TypeError, match=r"\['database'\]"):
            FolderPlatform('results', database=None)

    def test_interpreter_base_with_variadic_arguments_accepts_only_its_named_arguments(self):
        class OpenInterpreter(Interpreter):
            FLAGS = frozenset({PRICE})

            def __init__(self, parent_dataset=None, prices=None, *args, **options):
                super().__init__(parent_dataset=parent_dataset, prices=prices)
                self.options = options

            def _fetch(self, flag, effective_config, **kwargs):
                return self.prices

        class OpenPlatform(Platform):
            @classmethod
            def get_child_dataset_type(cls):
                return OpenInterpreter

        OpenPlatform.register_interpreter(OpenInterpreter)
        given = prices()
        interpreter = OpenPlatform(prices=given).get_interpreter_instance(OpenInterpreter)
        assert interpreter.prices is given and interpreter.options == {}
        with pytest.raises(
            TypeError,
            match=r"\['scale'\]. Accepted by PlatformDataset.__init__: \['attributes', 'cache', 'config', 'name', 'prices'\]",
        ):
            OpenPlatform(prices=given, scale=2)
