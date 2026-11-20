"""Illustrative demo taxonomy: this is not an official shipping-event standard."""
LABEL_DESCRIPTIONS = {
    "loading": "LNG cargo received by a vessel at an export terminal",
    "departure": "A carrier leaves a port or terminal",
    "arrival": "A carrier reaches a port, terminal or anchorage",
    "discharge": "LNG cargo is unloaded at a receiving facility",
    "ship_to_ship_transfer": "Cargo transferred between two vessels offshore",
    "route_diversion": "A vessel changes intended route or destination",
    "delay": "Voyage or berth schedule interrupted or postponed",
    "sanctions": "Official sanctions designation or restrictive measure",
    "detention": "A maritime authority holds or seizes a vessel",
    "vessel_status": "Location, AIS signal or navigational status update",
}
LABELS = tuple(LABEL_DESCRIPTIONS)
