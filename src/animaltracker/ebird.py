"""eBird API client for seasonal species filtering.

Fetches recent bird observations from eBird to filter detections
based on what species are actually being seen in your area.

Data provided by eBird (https://ebird.org), a project of the Cornell Lab of Ornithology.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

LOGGER = logging.getLogger(__name__)

# eBird API base URL
EBIRD_API_BASE = "https://api.ebird.org/v2"

# Cache duration in seconds (24 hours)
DEFAULT_CACHE_DURATION = 86400


@dataclass
class EBirdSpecies:
    """A species observed in eBird."""
    species_code: str
    common_name: str
    scientific_name: str
    observation_count: int = 0


@dataclass
class EBirdCache:
    """Cached eBird data."""
    species: Dict[str, EBirdSpecies] = field(default_factory=dict)
    scientific_names: Set[str] = field(default_factory=set)
    common_names: Set[str] = field(default_factory=set)
    last_updated: float = 0.0
    region: str = ""


class EBirdClient:
    """Client for fetching recent observations from eBird API."""
    
    def __init__(
        self,
        api_key: str,
        region: str = "US-MN",
        days_back: int = 14,
        cache_dir: Optional[Path] = None,
        cache_duration: int = DEFAULT_CACHE_DURATION,
        enabled: bool = True,
    ):
        """Initialize the eBird client.
        
        Args:
            api_key: eBird API key (get from https://ebird.org/api/keygen)
            region: Region code (e.g., "US-MN" for Minnesota, "US" for all USA)
            days_back: Number of days back to fetch observations (1-30)
            cache_dir: Directory to cache species list
            cache_duration: How long to cache data in seconds (default 24 hours)
            enabled: Whether eBird filtering is enabled
        """
        self.api_key = api_key
        self.region = region
        self.days_back = min(max(days_back, 1), 30)  # Clamp to 1-30
        self.cache_dir = cache_dir
        self.cache_duration = cache_duration
        self.enabled = enabled
        
        self._cache = EBirdCache()
        
        if cache_dir:
            self._cache_file = cache_dir / "ebird_cache.json"
            self._load_cache()
    
    def _load_cache(self) -> None:
        """Load cached species data from disk."""
        if not self._cache_file or not self._cache_file.exists():
            return
        
        try:
            with self._cache_file.open('r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._cache.region = data.get('region', '')
            self._cache.last_updated = data.get('last_updated', 0.0)
            self._cache.scientific_names = set(data.get('scientific_names', []))
            self._cache.common_names = set(data.get('common_names', []))
            
            # Rebuild species dict
            for sp_data in data.get('species', []):
                sp = EBirdSpecies(
                    species_code=sp_data['species_code'],
                    common_name=sp_data['common_name'],
                    scientific_name=sp_data['scientific_name'],
                    observation_count=sp_data.get('observation_count', 0),
                )
                self._cache.species[sp.scientific_name.lower()] = sp
            
            LOGGER.info("Loaded eBird cache: %d species for %s", 
                       len(self._cache.species), self._cache.region)
        except Exception as e:
            LOGGER.warning("Failed to load eBird cache: %s", e)
    
    def _save_cache(self) -> None:
        """Save species data to disk cache."""
        if not self._cache_file:
            return
        
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'region': self._cache.region,
                'last_updated': self._cache.last_updated,
                'scientific_names': list(self._cache.scientific_names),
                'common_names': list(self._cache.common_names),
                'species': [
                    {
                        'species_code': sp.species_code,
                        'common_name': sp.common_name,
                        'scientific_name': sp.scientific_name,
                        'observation_count': sp.observation_count,
                    }
                    for sp in self._cache.species.values()
                ],
            }
            
            with self._cache_file.open('w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            LOGGER.debug("Saved eBird cache: %d species", len(self._cache.species))
        except Exception as e:
            LOGGER.warning("Failed to save eBird cache: %s", e)
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if not self._cache.species:
            return False
        if self._cache.region != self.region:
            return False
        if time.time() - self._cache.last_updated > self.cache_duration:
            return False
        return True
    
    def refresh(self, force: bool = False) -> bool:
        """Refresh the species list from eBird API.
        
        Args:
            force: Force refresh even if cache is valid
            
        Returns:
            True if refresh was successful
        """
        if not self.enabled:
            return False
        
        if not self.api_key:
            LOGGER.warning("eBird API key not configured")
            return False
        
        if not force and self._is_cache_valid():
            LOGGER.debug("eBird cache is still valid")
            return True
        
        LOGGER.info("Fetching recent observations from eBird for %s (last %d days)",
                   self.region, self.days_back)
        
        try:
            url = f"{EBIRD_API_BASE}/data/obs/{self.region}/recent?back={self.days_back}"
            
            req = Request(url)
            req.add_header("X-eBirdApiToken", self.api_key)
            
            with urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            # Process observations
            species_map: Dict[str, EBirdSpecies] = {}
            
            for obs in data:
                sci_name = obs.get('sciName', '').lower()
                if not sci_name:
                    continue
                
                if sci_name not in species_map:
                    species_map[sci_name] = EBirdSpecies(
                        species_code=obs.get('speciesCode', ''),
                        common_name=obs.get('comName', ''),
                        scientific_name=obs.get('sciName', ''),
                        observation_count=1,
                    )
                else:
                    species_map[sci_name].observation_count += 1
            
            # Update cache
            self._cache.species = species_map
            self._cache.scientific_names = set(species_map.keys())
            self._cache.common_names = {sp.common_name.lower() for sp in species_map.values()}
            self._cache.last_updated = time.time()
            self._cache.region = self.region
            
            self._save_cache()
            
            LOGGER.info("eBird refresh complete: %d species currently being observed in %s",
                       len(species_map), self.region)
            
            return True
            
        except HTTPError as e:
            LOGGER.error("eBird API error: %s %s", e.code, e.reason)
            return False
        except URLError as e:
            LOGGER.error("eBird connection error: %s", e.reason)
            return False
        except Exception as e:
            LOGGER.error("eBird refresh failed: %s", e)
            return False
    
    def is_species_present(self, species_name: str) -> Optional[bool]:
        """Check if a species is currently being observed in the region.
        
        Args:
            species_name: Scientific name, common name, or SpeciesNet label
            
        Returns:
            True if species is present, False if not, None if can't determine
            (e.g., not a bird, eBird disabled, or no data)
        """
        if not self.enabled:
            return None
        
        if not self._cache.species:
            # Try to refresh if we have no data
            if not self.refresh():
                return None
        
        # Normalize the species name
        name_lower = species_name.lower().strip()
        
        # Handle SpeciesNet taxonomy format (e.g., "aves;passeriformes;...")
        if ';' in name_lower:
            parts = [p.strip() for p in name_lower.split(';')]
            # Check if this is a bird (aves)
            if 'aves' not in parts:
                return None  # Not a bird, eBird doesn't apply
            # Use the most specific part (last non-empty)
            for part in reversed(parts):
                if part and part not in ('aves', 'bird', 'animal'):
                    name_lower = part.replace('_', ' ')
                    break
        
        # Replace underscores with spaces
        name_lower = name_lower.replace('_', ' ')
        
        # Check scientific names
        if name_lower in self._cache.scientific_names:
            return True
        
        # Check common names
        if name_lower in self._cache.common_names:
            return True
        
        # Check partial matches (genus level)
        # e.g., "cardinalis" should match "cardinalis cardinalis"
        for sci_name in self._cache.scientific_names:
            if name_lower in sci_name or sci_name.startswith(name_lower + ' '):
                return True
        
        # Check if it's a generic bird term (don't filter these)
        generic_terms = {'bird', 'aves', 'passeriformes', 'passerine'}
        if name_lower in generic_terms:
            return None  # Can't determine for generic terms
        
        # Species not found in recent observations
        return False
    
    def get_species_info(self, species_name: str) -> Optional[EBirdSpecies]:
        """Get eBird info for a species if available."""
        if not self.enabled or not self._cache.species:
            return None
        
        name_lower = species_name.lower().strip().replace('_', ' ')
        
        # Handle taxonomy format
        if ';' in name_lower:
            parts = [p.strip() for p in name_lower.split(';')]
            for part in reversed(parts):
                if part and part not in ('aves', 'bird', 'animal'):
                    name_lower = part.replace('_', ' ')
                    break
        
        # Direct match on scientific name
        if name_lower in self._cache.species:
            return self._cache.species[name_lower]
        
        # Search by common name
        for sp in self._cache.species.values():
            if sp.common_name.lower() == name_lower:
                return sp
        
        return None
    
    def get_all_species(self) -> List[EBirdSpecies]:
        """Get all species currently in the cache."""
        return list(self._cache.species.values())
    
    @property
    def species_count(self) -> int:
        """Number of species in the cache."""
        return len(self._cache.species)
    
    @property
    def last_updated(self) -> float:
        """Timestamp of last update."""
        return self._cache.last_updated


def create_ebird_client(
    api_key: Optional[str] = None,
    api_key_env: str = "EBIRD_API_KEY",
    region: str = "US-MN",
    days_back: int = 14,
    cache_dir: Optional[Path] = None,
    cache_hours: int = 24,
    enabled: bool = True,
) -> Optional[EBirdClient]:
    """Create an eBird client if configured.
    
    Args:
        api_key: eBird API key (if not provided, will try environment variable)
        api_key_env: Environment variable name for API key
        region: Region code
        days_back: Days of observations to fetch
        cache_dir: Cache directory
        cache_hours: Cache duration in hours
        enabled: Whether eBird filtering is enabled
        
    Returns:
        EBirdClient if enabled and API key provided, None otherwise
    """
    import os
    
    if not enabled:
        LOGGER.info("eBird filtering is disabled")
        return None
    
    # Get API key from environment if not provided directly
    if not api_key:
        api_key = os.environ.get(api_key_env, "")
    
    if not api_key:
        LOGGER.info("eBird API key not configured (%s not set), seasonal filtering disabled", api_key_env)
        return None
    
    # Convert cache hours to seconds
    cache_duration = cache_hours * 3600
    
    client = EBirdClient(
        api_key=api_key,
        region=region,
        days_back=days_back,
        cache_dir=cache_dir,
        cache_duration=cache_duration,
        enabled=enabled,
    )
    
    # Do initial refresh
    client.refresh()
    
    return client
