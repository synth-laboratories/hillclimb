"""Declarative Temper panels shared by the standard board and optional lenses."""


def table(name, source, path, columns):
    return {
        "kind": "table",
        "title": name,
        "from": {
            "source": {
                "input": source,
                "path": path,
                "transform": [{"op": "limit", "count": 400}],
            },
            "columns": [{"header": label, "field": key} for label, key in columns],
        },
    }


def plot(kind, title, deck, path, **fields):
    return {
        "kind": kind,
        "title": title,
        "from": {
            "source": {
                "input": deck,
                "path": path,
                "transform": [{"op": "limit", "count": 400}],
            },
            **fields,
        },
    }
