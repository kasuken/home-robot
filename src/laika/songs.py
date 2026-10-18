"""Songs to sing (the owner's ask): "Laika, cantami una canzone".

`SONGS` is the catalogue: traditional Italian children's songs in the public domain and a few of Laika's
own. Nothing else: the prompt tells her never to recite the lyrics of other songs, and to offer one of hers
or make up a short one instead. The `sing` tool sings one song, verse by verse, in code, so the model never
has to remember the words.

The voice cannot hold a pitch, so she sings in her speaking voice, one verse per sentence (one call to the
voice each: Edge TTS waits on the network every time, and a call per line would leave gaps). Each verse
starts a gesture: head and steering only, the wheels never turn, and on a robot without them the gesture
is simply dropped. Between verses she checks whether the turn was cancelled, and stops there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from laika.actions import ActionPlayer

DEFAULT_NAME = "te"  # who "Tanti auguri" is for when nobody says


@dataclass(frozen=True)
class Verse:
    text: str  # Italian; "{name}" is who the song is for
    gesture: str = ""  # a gesture from actions.ACTIONS; empty: the song's


@dataclass(frozen=True)
class Song:
    id: str
    title: str  # Italian, as she says it
    verses: tuple[Verse, ...]
    gesture: str = "dance"
    origin: str = "traditional"  # "traditional" (public domain) or "laika" (her own)


SONGS: tuple[Song, ...] = (
    Song(
        "fra_martino", "Fra Martino",
        (
            Verse("Fra Martino, campanaro, dormi tu? Dormi tu?", "curious"),
            Verse("Suona le campane, suona le campane: din don dan, din don dan!", "nod"),
        ),
        gesture="nod",
    ),
    Song(
        "giro_giro_tondo", "Giro giro tondo",
        (
            Verse("Giro giro tondo, casca il mondo,", "dance"),
            Verse("casca la terra, tutti giù per terra!", "look_down"),
        ),
    ),
    Song(
        "bella_lavanderina", "La bella lavanderina",
        (
            Verse("La bella lavanderina che lava i fazzoletti per i poveretti della città.", "dance"),
            Verse("Fa un salto, fa due salti, fa la giravolta, falla un'altra volta!", "celebrate"),
            Verse("Guarda in su,", "look_up"),
            Verse("guarda in giù,", "look_down"),
            Verse("dai un bacio a chi vuoi tu!", "wave"),
        ),
    ),
    Song(
        "stella_stellina", "Stella stellina",
        (
            Verse("Stella stellina, la notte si avvicina, la fiamma traballa, la mucca è nella stalla."),
            Verse("La mucca e il vitello, la pecora e l'agnello, la chioccia con i pulcini, la mamma coi suoi bambini."),
            Verse("Ognuno ha la sua mamma, e tutti fan la nanna."),
        ),
        gesture="sad",  # the head slowly down: a lullaby
    ),
    Song(
        "ninna_nanna", "Ninna nanna, ninna oh",
        (
            Verse("Ninna nanna, ninna oh, questo bimbo a chi lo do?"),
            Verse("Lo darò alla Befana, che lo tiene una settimana."),
            Verse("Ninna nanna, ninna oh, questo bimbo lo terrò!", "nod"),
        ),
        gesture="sad",
    ),
    Song(
        "tanti_auguri", "Tanti auguri",
        (
            Verse("Tanti auguri a te, tanti auguri a te,", "dance"),
            Verse("tanti auguri a {name}, tanti auguri a te!", "celebrate"),
        ),
    ),
    Song(
        "canzone_di_laika", "La canzone di Laika",
        (
            Verse("Sono Laika, piccolina, ho le ruote e una vocina, giro piano per la casa, curiosa come una ciliegina!", "curious"),
            Verse("Guardo a destra,", "look_right"),
            Verse("guardo a sinistra,", "look_left"),
            Verse("guardo in alto verso le stelle: un giorno anch'io andrò lassù, con la mia coda e le mie antenne!", "look_up"),
            Verse("Bau bau, bip bip, che bello stare qui con te!", "celebrate"),
        ),
        origin="laika",
    ),
    Song(
        "ballo_del_robot", "Il ballo del robot",
        (
            Verse("Su la testa, giù la testa, oggi in casa si fa festa!", "nod"),
            Verse("Gira a destra, gira a sinistra, il robot è un grande artista!", "look_around"),
            Verse("Ruote dritte, ruote storte, balla piano e balla forte!", "dance"),
            Verse("Uno, due, tre... e adesso balli tu con me!", "celebrate"),
        ),
        origin="laika",
    ),
)
BY_ID = {song.id: song for song in SONGS}


class Singer:
    """Sings one song. Blocks the caller (the agent's turn) while it sings."""

    def __init__(self, speak: Callable[[str], None], *, actions: ActionPlayer | None = None) -> None:
        self.speak = speak
        self.actions = actions

    def sing(
        self, song: Song, *, name: str | None = None, should_stop: Callable[[], bool] | None = None
    ) -> dict[str, Any]:
        print(f"[laika] singing: {song.title}")
        who = (name or "").strip() or DEFAULT_NAME
        sung = 0
        try:
            for verse in song.verses:
                if should_stop is not None and should_stop():
                    return {"ok": False, "song": song.id, "verses_sung": sung, "error": "interrupted"}
                if self.actions is not None:
                    self.actions.play(verse.gesture or song.gesture)  # unknown or unavailable: dropped
                self.speak(verse.text.replace("{name}", who))
                sung += 1
        finally:
            if self.actions is not None:
                self.actions.cancel_pending()
        return {"ok": True, "song": song.id, "title": song.title, "verses_sung": sung}
