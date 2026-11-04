"""Country hints restrict symbol style; a hint can never create WALK evidence."""
import pytest

from streetsense.config import Config
from streetsense.traffic_light import (
    LightReading, apply_signal_profile, classify_light, signal_profile,
)
from test_eu_figure import green_man_head


@pytest.mark.parametrize('country,expected', [('GB', 'green_figure'), ('uk', 'green_figure'),
    ('DE', 'green_figure'), ('EU', 'green_figure'), ('US', 'white_person'), ('CA', 'white_person'),
    ('', 'auto'), ('unverified', 'auto'), (None, 'auto')])
def test_explicit_country_is_a_style_hint(country, expected):
    assert signal_profile(country) == expected


def test_country_config_reads_environment_at_construction(monkeypatch):
    monkeypatch.setenv('STREETSENSE_SIGNAL_COUNTRY', ' gb ')
    assert Config().reader.country == 'GB'


def test_explicit_country_overrides_environment_for_local_commands(monkeypatch):
    from argparse import Namespace
    from streetsense.cli import _local_config

    monkeypatch.setenv('STREETSENSE_SIGNAL_COUNTRY', 'US')
    assert _local_config(Namespace(signal_country='GB')).reader.country == 'GB'


def test_serve_command_accepts_country_hint(monkeypatch):
    from streetsense import cli

    seen = []
    monkeypatch.setattr(cli, 'cmd_serve', lambda args: seen.append(args.signal_country) or 0)
    assert cli.main(['serve', '--signal-country', 'gb']) == 0
    assert seen == ['GB']


@pytest.mark.parametrize('state', ['unknown', 'red', 'yellow', 'green', 'countdown'])
@pytest.mark.parametrize('profile', ['auto', 'green_figure', 'white_person'])
def test_country_hint_never_creates_walk(state, profile):
    reading = LightReading(state, .9, .1, .75)
    assert apply_signal_profile(reading, green_man_head(), profile) == reading


def test_uk_profile_keeps_green_figure_and_rejects_white_only_walk():
    reading = LightReading('walk', .9, .1, .75, figure=True)
    assert apply_signal_profile(reading, green_man_head(), 'green_figure').state == 'walk'
    white = green_man_head(colour=(255, 255, 255))
    assert apply_signal_profile(reading, white, 'green_figure').state == 'unknown'
    assert apply_signal_profile(reading, white, 'auto').state == 'walk'


def test_hand_profile_does_not_accept_green_figure_as_white_person():
    reading = classify_light(green_man_head())
    assert reading.state == 'walk'
    assert apply_signal_profile(reading, green_man_head(), 'white_person').state == 'unknown'


def test_profile_preserves_pale_green_halo():
    reading = LightReading('walk', .9, .1, .75, figure=True)
    assert apply_signal_profile(reading, green_man_head(colour=(180, 230, 175)), 'green_figure').state == 'walk'
